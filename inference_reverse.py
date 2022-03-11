#!/usr/bin/env python
# coding: utf-8
# Two separate models for artifact classification and HFO with spike classification
###
import torch
import torch.nn as nn
from random import random
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from sklearn.metrics import f1_score
import sys
import random

from src.dataloader import create_patient_loader
from src.dataloader_spike import create_patient_loader_90
from src.utilities import *
from src.config import inference_parse
from src.meter import TrainingMeter, Meter
from src.model import NeuralCNN
from patient_info import seizure_free_patient_names

def validate(val_loader, model, criterion, computing_device, long_eeg_flag, fn=None):
    start = time.time()
    meter_s = Meter("spike")
    model_s = model["spike"]
    model_s.eval()
    for _ , (image, waveform, intensity, label, info, start_end) in enumerate(val_loader, 0):

        with torch.no_grad():
            _ , s_, train_s = create_sa_labels(image, waveform, intensity, label, info, start_end, computing_device)
            if not train_s:
                print(label.sum())
                continue
            outputs_s = model_s(s_["inputs"]).squeeze()
            if not long_eeg_flag:
                s_["label"] = s_["label"].squeeze()[:,0] ## change it to binary #90 inference 10 comment it !!! 90 inference 90 hold
            if outputs_s.dim() == 0:
                outputs_s = outputs_s.unsqueeze(0)
                s_["label"] = s_["label"].unsqueeze(0)
            #print(outputs_s.shape, s_["label"].shape)
            loss_s = criterion(outputs_s, s_["label"])
            if not fn:
                meter_s.update_loss(loss_s.detach().cpu().numpy())
                meter_s.update_outputs(outputs_s.detach().cpu().numpy(), s_[
                                       "label"].cpu().numpy())
            else:
                meter_s.add(s_["spectrum"],
                            s_["label"].detach().cpu().numpy(),
                            s_["channel_name"],
                            s_["start_end"],
                            s_["intensity"],
                            s_["waveform"],
                            outputs_s.detach().cpu().numpy())
    acc_s = meter_s.accuracy()
    if fn is not None:
        loss_s = 0
        meter_s.dump_prediction(os.path.join(fn, f"spikes.pkl"))
        #meter_s.dump_pickle(os.path.join(fn, f"spikes.pkl"))
    else:         
        loss_s = meter_s.loss()
    f1_s = meter_s.f1()
    print('Inference: Time %.3f, loss_s: %.3f, accuracy_s: %.3f , f1_s: %0.3f' %
          (time.time()-start, loss_s, acc_s, f1_s))

    return loss_s, acc_s, f1_s


def create_sa_labels(spectrum, waveform, intensity, label, channel_name, start_end, computing_device):
    channel_name = np.array(channel_name)
    label = label.squeeze().float()
    s_ ={}    
    #s_spectrum_norm = torch.log(spectrum)
    inputs_s = torch.stack([normalize_img(spectrum),waveform,normalize_img(intensity)], dim=1, out=None).to(computing_device).float()
    label_s= label.to(computing_device)
    #print(inputs_s.shape)
    s_ = {
        "inputs": expand_dim(inputs_s, 4),
        "spectrum":spectrum,
        "label": expand_dim(label_s, 1), 
        "intensity":intensity,
        "waveform":waveform,
        "channel_name":channel_name, 
        "start_end":start_end
    }
    
    return None , s_ , True


def pipeline(args, patient_name):
    data_dir = args.data_dir
    res_dir = os.path.join(args.work_dir, args.res_dir)
    trained_model_path = ""
    model_dir = os.path.join(args.work_dir, args.model_dir)
    for fn in os.listdir(model_dir):
        if fn.endswith("model_s.pth"):
            trained_model_path= os.path.join(model_dir, fn)

    """
    Modified from train_reverse.py:
    Load the trained model and test on the training patient
    """

    model_spike = NeuralCNN(num_classes=2)

    batch_size = 512         # Number of samples in each minibatch
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        print("cuda is supported")
        computing_device = torch.device(args.device)
        extras = {"num_workers": 1, "pin_memory": True}
    else:  # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        
    model_spike = model_spike.to(computing_device)
    model = {"spike": model_spike}
    
    start_time = time.time()
    if not args.long_eeg:
        test_loader = create_patient_loader(data_dir, patient_name,batch_size,shuffle=False,extras=extras)
    else:
        test_loader = create_patient_loader_90(data_dir, patient_name,batch_size,shuffle=False,extras=extras)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Prepare dataset | Time: {epoch_mins}m {epoch_secs}s')

    criterion = nn.BCELoss(reduction="none").to(computing_device)

    print("-----------------testing ----------------")

    if trained_model_path is not None:
        model['spike'].load_state_dict(torch.load(trained_model_path)['state_dict'])
        model['spike'].eval()

    print("patient_names is", patient_name)
    stats_folder = os.path.join(res_dir, patient_name)
    clean_folder(stats_folder)
    loss_s, acc_s, _ = validate(test_loader, model, criterion,
                                 computing_device, args.long_eeg ,fn=stats_folder)
    return loss_s, acc_s


def patient_fold(args):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)
    clean_folder(args.res_dir)
    
    p_names = os.listdir(args.data_dir)
    if args.long_eeg:
        p_names = list(set(p_names) - set(seizure_free_patient_names))
    p_names = os.listdir(args.data_dir)
    for p_name in p_names:
        loss_s, acc_s = pipeline(args, p_name)


if __name__ == "__main__":
    args = inference_parse(sys.argv[1:])
    print(args)
    patient_fold(args)
