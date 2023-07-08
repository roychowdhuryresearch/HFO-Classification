
import os, time, copy, sys

import torch
import torch.nn as nn
import torch.optim as optim
from random import random, sample
import numpy as np
from torch.utils.data import dataset, Subset, WeightedRandomSampler

from src.dataloader import HFODataset, read_data, parallel_process, SubsetRandomSampler
from src.model import NeuralCNN, PreProcessing
from src.meter import TrainingMeter, Meter
from src.training_utils import *
import torch.utils.data as data
import random
import torch
from torch.utils.data import  DataLoader
from sklearn.model_selection import  KFold
from src.args import args
import copy
from trainer import Trainer
from src.model import NeuralCNN, PreProcessing

from pathlib import Path
import torch
from torchvision.models import resnet18
import torch_pruning as tp
from thop import clever_format
import pandas as pd

def get_pruner(model,iterative_steps, example_inputs):
    imp = tp.importance.MagnitudeImportance(p=2)

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1:
            ignored_layers.append(m) # DO NOT prune the final classifier!

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    return model, pruner, base_macs, base_nparams

def pruning_pipeline(model_folder, args, iterative_steps, fintune_freq, device):
    model_path = os.path.join(model_folder, "model_best.tar")
    fold = int(model_folder.split("/")[-1].split("_")[-1])
    res_dir = Path(model_folder).parent.parent
    n_fold = len([f for f in os.listdir(os.path.join(res_dir, "ckpt")) if "fold" in f])

    ckpt = torch.load(model_path, map_location="cpu")
    model = ckpt["model"].to(device)
    preprocessing_dict = ckpt["preprocessing"]
    preprocessing = PreProcessing.from_dict(preprocessing_dict)
    model_input_w = preprocessing.crop_index_w
    model_input_h = preprocessing.crop_index_h
    in_channels = model.in_channels
    example_inputs =  torch.randn(1, in_channels, model_input_h, model_input_w).to(device)
    model, pruner, base_macs, base_nparams = get_pruner(model, iterative_steps, example_inputs)
    args["save_checkpoint"] = False
    args["res_dir"] = res_dir
    trainer = Trainer(args, verbose=False)
    dfs = []
    for i in range(iterative_steps):
        pruner.step()
        if i % fintune_freq == 0:
            macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
            macs_str, _ = clever_format([macs, nparams], "%.3f")
            model_trained, acc = trainer.onefold_crossvalidation(n_fold, fold, model)
            print(f"Prune Step {i+1}/{iterative_steps}: Acc:{acc:.3f}, MACs:{macs_str}, {macs/base_macs:.2f}x MACs, {nparams/base_nparams:.2f}x Params (ResNet18)")
            dfs.append(pd.DataFrame({"Step":i, "Acc":acc,  "MACs":macs_str,"Params": nparams}, index=[0]))
            model.load_state_dict(model_trained.state_dict())
            ckpt_save = {
                "model": model,
                "preprocessing": preprocessing.to_dict(),
                "args": args,
            }   
            torch.save(ckpt_save, os.path.join(model_folder, f"pruned_model.tar"))
    df = pd.concat(dfs)
    df.to_csv(os.path.join(model_folder, f"pruning_stats.csv"))
    return model

if __name__ == "__main__":
    device = sys.argv[1]
    iterative_steps = 5000
    args['device'] = device
    args["num_epochs"] = 5   # 5 epochs for each pruning step
    fintune_freq = 500   # fintune the model every 500 pruning steps
    
    args["data_dir"] = "data_training/artifact_data"
    model_folder = "result/artifact_data_win285_freq10_300_shift50/ckpt/fold_0"
    pruning_pipeline(model_folder, args, iterative_steps, fintune_freq, device)