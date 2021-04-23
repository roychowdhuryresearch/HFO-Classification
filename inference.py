
import os, sys, time, copy
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from random import random
import random
import numpy as np

from src.utilities import *
from src.dataloader import create_patient_eliminate_loader, HFODataset
from src.model import NeuralCNN
from src.config import inference_parse
from src.meter import InferenceStats


def create_testing_loader(data_folder, patient_name):
    # once all single json datasets are created you can concat them into a single one:
    hfo_dataset = HFODataset(data_dir=data_folder, patient_name=patient_name)
    data_loader = DataLoader(hfo_dataset, batch_size=512, num_workers=1, pin_memory=True)
    return data_loader

def run_model(model_a, model_s, iterator, computing_device):
    stats = InferenceStats()
    for i, (spectrum, waveform, intensity, _ ,info, start_end) in enumerate(iterator, 0):
        channel_name = np.array(info)
        #spectrum_norm = normalize_img(spectrum)
        spectrum_norm = spectrum
        inputs_a = torch.stack([spectrum_norm,spectrum_norm,spectrum_norm], dim=1, out=None).to(computing_device).float()  

        #s_spectrum_norm = normalize_img(torch.log(spectrum))
        s_spectrum_norm = spectrum
        inputs_s = torch.stack([s_spectrum_norm,waveform,intensity], dim=1, out=None).to(computing_device).float()

        with torch.no_grad():       
            outputs_a = model_a(inputs_a).detach().cpu().numpy()
            outputs_s = model_s(inputs_s).detach().cpu().numpy()
            stats.add(outputs_a, outputs_s, channel_name, start_end)        
    return stats


def inference( data_dir, res_folder, model_folder, computing_device):
    path_artifacts = ""
    path_spike = ""
    for fn in os.listdir(model_folder):
        if fn.endswith("model_a.pth"):
            path_artifacts = os.path.join(model_folder, fn)
        if fn.endswith("model_s.pth"):
            path_spike = os.path.join(model_folder, fn)
    
    model_artifact = NeuralCNN(num_classes=2).to(computing_device)
    model_spike = NeuralCNN(num_classes=2).to(computing_device)
    
    model_artifact.load_state_dict(torch.load(path_artifacts)["state_dict"])
    model_spike.load_state_dict(torch.load(path_spike)["state_dict"])
    
    model_artifact.eval()
    model_spike.eval()

    patient_list = sorted(os.listdir(data_dir))

    for p in patient_list:
        print("working on ", p)
        data_loader = create_testing_loader(data_dir, p)
        p_stats = run_model(model_artifact, model_spike, data_loader, computing_device)
        fn = os.path.join(res_folder, p+".csv")
        p_stats.export_cvs(fn)

if __name__ == "__main__":
    
    args = inference_parse(sys.argv[1:])
    res_dir = os.path.join(args.work_dir, args.res_dir)
    clean_folder(res_dir)
    model_dir = os.path.join(args.work_dir, args.model_dir)
    data_dir = args.data_dir
    computing_device = torch.device(args.device)
    inference( data_dir, res_dir,model_dir, computing_device)
    
    
