import os, time, copy, sys

import torch
import torch.nn as nn
import torch.optim as optim
from random import random
import numpy as np

from src.utilities import *
from src.dataloader_ecHFO import HFODataset
from src.model import NeuralCNN
from src.config import arg_parse90
from src.meter import TrainingMeter, Meter
from src.training_utils import *
from patient_info import seizure_free_patient_names, patient_90
import torch.utils.data as data
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler


class Inference():
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.mode = args.model_mode
        self.res_dir = os.path.join(args.work_dir, args.res_dir, self.mode, "ckpt")
        self.batch_size = 128
        self.patient_res_folder = None
        self.device = args.device  
        print(self.device)
        self.criterion = nn.BCELoss(reduction="none").to(self.device)
        self.seed = args.seed
    
    def __initialize_model(self, model_path = None):
        if model_path is None:
            model_path = self.patient_res_folder
        model = NeuralCNN(num_classes=2, freeze_layers=True, dropout_p=0)
        model.load_state_dict(torch.load(os.path.join(model_path, "model_s.pth"), map_location = self.device)['state_dict'])
        model = model.to(self.device)
        return model

    def __construct_test_loader(self, patient_name):
        test_set = HFODataset(data_dir=self.data_dir, patient_name=patient_name, transform=None, filter_both = False, keep= True, mode= self.mode)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, num_workers=1, 
                                pin_memory=True)
        return test_loader


    def __create_sa_labels(self, image, waveform, intensity, label, channel_name, start_end):
            channel_name = np.array(channel_name)
            label = label.squeeze().float()
            inputs_s = (
                torch.stack([image, waveform, intensity], dim=1, out=None)
                .to(self.device)
                .float()
            )
            label_s = label.to(self.device)
            s_ = {
                "inputs": expand_dim(inputs_s, 4),
                "spectrum": image,
                "label": expand_dim(label_s, 1).squeeze(),
                "intensity": intensity,
                "waveform": waveform,
                "channel_name": channel_name,
                "start_end": start_end
            }
            return s_

    def __inference(self, model, loader, fn=None, meter_overall=None):
        start = time.time()
        meter_s = Meter("spike")
        model_s = model
        model_s.eval()
        for _, (pt_name, image, waveform, intensity, label, info, start_end) in enumerate(loader, 0):
            with torch.no_grad():
                s_ = self.__create_sa_labels(
                    image, waveform, intensity, label, info, start_end)

                outputs_s = model_s(s_["inputs"]).squeeze()
        
                if outputs_s.dim() == 0:
                    outputs_s = outputs_s.unsqueeze(0)
                    s_["label"] = s_["label"].unsqueeze(0)
                
                behavior_labels = s_["label"][:, 0].cpu()
                outputs_s = outputs_s.detach().cpu()
                loss_s = self.criterion(outputs_s, behavior_labels)
                
                if not fn:
                    meter_s.update_loss(loss_s.numpy())
                    meter_s.update_outputs(outputs_s.numpy(), behavior_labels.numpy())
                else:
                    meter_s.add(
                        s_["spectrum"],
                        s_["label"].detach().cpu().numpy()[:, 0],
                        s_["channel_name"],
                        s_["start_end"],
                        s_["intensity"],
                        s_["waveform"],
                        outputs_s.numpy(),
                        pt_name= pt_name
                    )

                # meter_overall
                if meter_overall is not None:
                    meter_overall.update_loss(loss_s.numpy())
                    meter_overall.update_outputs(outputs_s.numpy(), behavior_labels.numpy())

      
        if fn is not None:
            loss_s, f1_s, acc_s = 0,0,0
            meter_s.dump_csv(os.path.join(fn, "inference.csv")) ###
        else:
            loss_s = meter_s.loss()
            f1_s = meter_s.f1()
            acc_s = meter_s.accuracy()
        print(
            "Inference: Time %.3f, loss_s: %.3f, accuracy_s: %.3f , f1_s: %0.3f"
            % (time.time() - start, loss_s, acc_s, f1_s)
        )

        return loss_s, acc_s, f1_s

    def patientwise_crossvalidation(self):
        meter_overall = Meter("spike")
        accuracies = []

        for p in patient_90:
            self.patient_res_folder = os.path.join(self.res_dir, p)
            loader = self.__construct_test_loader(p)
            model = self.__initialize_model()
            # self.__inference(model,loader, self.patient_res_folder)
            _, acc_s, _ = self.__inference(model,loader, self.patient_res_folder, meter_overall)
            accuracies.append([acc_s])
        acc_overall = meter_overall.accuracy()
        loss_overall = meter_overall.loss()
        f1_overall = meter_overall.f1()
        print(
            "Overall --- loss_s: %.3f, accuracy_s: %.3f , f1_s: %0.3f"
            % (loss_overall, acc_overall, f1_overall)
        )
        print("Mean accuracy: %.3f" % np.mean(accuracies))



    def kfold_crossvalidation(self):
        model_names = ["fold0", "fold1", "fold2", "fold3", "fold4"]
        for model_name in model_names:
            model_path = os.path.join(self.res_dir, model_name)
            for p in patient_90:
                self.patient_res_folder = os.path.join(model_path, p)
                os.makedirs(self.patient_res_folder, exist_ok=True)
                loader = self.__construct_test_loader(p)
                model = self.__initialize_model(model_path=model_path)
                _, acc_s, _ = self.__inference(model,loader, self.patient_res_folder)

        
if __name__ == "__main__":
    args = arg_parse90(sys.argv[1:])
    print(args)
    inference = Inference(args)
    inference.kfold_crossvalidation()