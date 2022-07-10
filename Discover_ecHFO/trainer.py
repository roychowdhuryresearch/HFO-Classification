
import os, time, copy, sys

import torch
import torch.nn as nn
import torch.optim as optim
from random import random, sample
import numpy as np
from torch.utils.data import dataset, Subset, WeightedRandomSampler

from src.utilities import *
from src.dataloader_ecHFO import HFODataset
from src.model import NeuralCNN
from src.config import arg_parse90
from src.meter import TrainingMeter, Meter
from src.training_utils import *
from patient_info import patient_names
import torch.utils.data as data
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

class Trainer():
    def __init__(self, args):

        self.data_dir = args.data_dir
        self.mode = args.model_mode
        self.res_dir = os.path.join(args.work_dir, args.res_dir, self.mode, "ckpt")  #
        self.num_epochs = args.num_epochs_s  # Number of full passes through the dataset
        self.batch_size = args.batch_size  # Number of samples in each minibatch
        self.learning_rate = args.learning_rate_s
        self.seed = args.seed  # Seed the random number generator for reproducibility
        self.p_val = args.p_val  # Percent of the overall dataset to reserve for validation
        self.p_test = args.p_test  # Percent of the overall dataset to reserve for testing
        self.device = args.device
        os.makedirs(self.res_dir, exist_ok=True)
        self.criterion = nn.BCELoss(reduction="none").to(self.device)
        ## process 
        self.all_labels = [] 
        self.list_of_datasets = self.__construct_dataset()
        self.all_ds = data.ConcatDataset(self.list_of_datasets)

    def kfold_crossvalidation(self, K):
        for i in range(K):
            print('------------', i, 'th fold ------------')
            train_loader, val_loader, test_loader = self.__construct_training_valid_set_kfold(K, i)
            res_dir = os.path.join(self.res_dir, "fold%d"%i)
            clean_folder(res_dir)
            model = self.train(train_loader, val_loader, res_dir)
            self.validate(test_loader, model, res_dir)


    def __initialize_model(self):
        return NeuralCNN(num_classes=2, freeze_layers=True, dropout_p=0).to(self.device)
    
    def __initialize_optimizer(self, model):
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate)

    def __construct_dataset(self):
        print("original_set")
        since = time.time()
        list_of_datasets = []

        for j in patient_names:
            p_dataset = HFODataset(data_dir=self.data_dir, patient_name=j, transform=None, filter_both=True, mode=self.mode)
            list_of_datasets.append(p_dataset)
            self.all_labels.append(p_dataset.labels)
        time_elapsed = time.time() - since
        print(
            "original_set in {:.0f}m {:.0f}s ".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        return list_of_datasets

    def __construct_training_valid_set_kfold(self, K, k):
        all_ds = self.all_ds
        dataset_size = len(all_ds)
        fakex = np.arange(dataset_size)
        all_labels = np.concatenate(self.all_labels)
        
        counter = 0
        sss = StratifiedKFold(n_splits=K, random_state=0,  shuffle=True)
        for train_index, test_index in sss.split(fakex, all_labels):
            ss = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=None)
            for t_index, valid_index in ss.split(fakex[train_index], all_labels[train_index]):
                real_train_index = train_index[t_index]
                real_valid_index = train_index[valid_index]
            if counter == k:
                break
            else:
                counter += 1

        num_pos_train = np.sum(all_labels[real_train_index])
        num_neg_train = len(real_train_index) - num_pos_train

        train_label = all_labels[real_train_index]
        samples_weight = np.zeros_like(train_label) + num_neg_train*1.0/(len(real_train_index))
        pos_index = np.where(train_label == 0)[0]
        samples_weight[pos_index] = num_pos_train*1.0/(len(real_train_index))
        
        sample_train = WeightedRandomSampler(samples_weight, len(samples_weight))
        sample_val = SubsetRandomSampler(real_valid_index)
        sample_test = SubsetRandomSampler(test_index)
        train_set = Subset(all_ds, real_train_index)
        train_loader = DataLoader(train_set, batch_size=self.batch_size,
                                sampler=sample_train, num_workers=1,
                                pin_memory=True)
        val_loader = DataLoader(all_ds, batch_size=self.batch_size,
                        sampler=sample_val, num_workers=1,
                            pin_memory=True)
        test_loader = DataLoader(all_ds, batch_size=self.batch_size, num_workers=1,
                                sampler=sample_test, pin_memory=True)

        return train_loader, val_loader, test_loader

    
    def __construct_test_loader(self, patient_name):
        test_set = None
        for ds in self.list_of_datasets:
            if ds.patient_name == patient_name: 
                test_set = ds
                break
        
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


    def train(self, train_loader, valid_loader, checkpoint_folder):
        train_meter_dir = os.path.join(checkpoint_folder, "train_meter")
        os.makedirs(train_meter_dir)

        model = self.__initialize_model()
        optimizer = self.__initialize_optimizer(model)
        train_meter = TrainingMeter("spike")
        since = time.time()
        best_acc, v_loss, v_acc, v_f1 = 0, 100, 0, 0
        best_model = None       
        for epoch in range(self.num_epochs):
            M = 1
            print("-" * 10)
            model.train()
            meter_s = Meter("spike")
            for _, (pt_name, image, waveform, intensity, label, info, start_end) in enumerate(train_loader, 0):
                s_ = self.__create_sa_labels(image, waveform, intensity, label, info, start_end)
                optimizer.zero_grad()
                outputs = model(s_["inputs"]).squeeze(1)
                loss_s = self.criterion(outputs, s_["label"][:, 0]) # bahavioral

                meter_s.update_loss(loss_s.detach().cpu().numpy())
                meter_s.add(
                        s_["spectrum"],
                        s_["label"].detach().cpu().numpy()[:, 0],
                        s_["channel_name"],
                        s_["start_end"],
                        s_["intensity"],
                        s_["waveform"],
                        outputs.detach().cpu().numpy(),
                        pt_name = pt_name,
                    )
                loss_s = torch.sum(loss_s) * 1.0 / len(outputs)
                loss_s.backward()
                optimizer.step()
            # Print the loss averaged over the last N mini-batches
            loss_s = meter_s.loss()
            acc_s = meter_s.accuracy()
            f1, recall, precision, b_acc = meter_s.f1()
            meter_s.dump_csv(os.path.join(train_meter_dir, "train_"+str(int(epoch)))+".csv")
            print("Epoch %d, loss_s: %.3f, acc: %.3f, b_acc: %.3f f1: %.3f, recall: %.3f, precision: %.3f" % (epoch + 1, loss_s, acc_s, b_acc ,f1, recall, precision))
            
            # Validation
            if epoch % M == 0 and epoch != 0:
                fn = os.path.join(checkpoint_folder, "validation")
                os.makedirs(fn, exist_ok=True)
                v_loss, v_acc, v_f1, b_acc = self.validate(valid_loader, model, fn = fn )
                best_acc, best_model = pick_best_model_acc(
                    model,
                    best_model,
                    epoch,
                    b_acc,
                    best_acc,
                    checkpoint_folder,
                    model_name="s"+str(epoch),
                )
            train_meter.add(acc_s, loss_s, v_loss, v_acc, 0, v_f1, 0, 0)
        print("Training complete after", epoch, "epochs")
        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s ".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        train_meter.dump_pickle(os.path.join(checkpoint_folder, "training_curve_s.pkl"))
        return best_model
    
    def validate(self, loader, model, fn = None):
        start = time.time()
        meter_s = Meter("spike")
        model_s = model
        model_s.eval()
        for _, (pt_name,image, waveform, intensity, label, info, start_end) in enumerate(loader, 0):
            with torch.no_grad():
                s_ = self.__create_sa_labels(
                    image, waveform, intensity, label, info, start_end)

                outputs_s = model_s(s_["inputs"]).squeeze()
                behavior_labels = s_["label"].squeeze()[:, 0].cpu()
                bad_labels = s_["label"].squeeze()[:, 1].cpu()
                
                if outputs_s.dim() == 0:
                    outputs_s = outputs_s.unsqueeze(0)
                    s_["label"] = s_["label"].unsqueeze(0)
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
                        pt_name = pt_name,
                    )
        acc_s = meter_s.accuracy()
        if fn is not None:
            loss_s = 0
            #meter_s.dump_pickle(os.path.join(fn, "spikes.pkl"))
            meter_s.dump_csv(os.path.join(fn, "spikes.csv"))
        else:
            loss_s = meter_s.loss()
        f1, recall, precision, b_acc = meter_s.f1()
        print(
            "Inference: Time %.3f, loss_s: %.3f, acc: %.3f, b_acc: %.3f , f1: %0.3f, recall: %0.3f, precision: %0.3f"
            % (time.time() - start, loss_s, acc_s, b_acc ,f1, recall, precision)
        )

        return loss_s, acc_s, f1, b_acc


if __name__ == "__main__":
    args = arg_parse90(sys.argv[1:])
    
    print(args)
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(args.seed)
    random.seed(args.seed)
    trainer = Trainer(args)
    trainer.kfold_crossvalidation(5)