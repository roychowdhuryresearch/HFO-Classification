
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
from src.args import data_args as feature_param
from feature_extraction import extract_features_perpatient
def inference_one_patient(data_path, pt_name, model, device, feature_param, fn, preprocessing):
    outfolder = data_path
    df_suffix = fn.split("/")[-1].split(".")[0]
    if not os.path.exists(os.path.join(outfolder, pt_name, fn)):
        print(f"no {fn} in {pt_name}")
        return
    feature_fn = os.path.join(outfolder, pt_name ,f"data_{df_suffix}.npz")
    if not os.path.exists(feature_fn):
        df = pd.read_csv(os.path.join(data_path, pt_name, fn))
        if "labels" not in df.columns:
            df["labels"] = np.zeros(len(df))
        df.to_csv(os.path.join(data_path, pt_name, fn), index=False)
        extract_features_perpatient(data_path, pt_name, feature_param, outfolder, fn, df_suffix)
    try:
        data = np.load(os.path.join(outfolder, pt_name , f"data_{df_suffix}.npz"), allow_pickle=True)
        feature = data["feature"]
        starts = data["starts"]/feature_param["resample"]
        ends = data["ends"]/feature_param["resample"]
        channel_names = data["channel_names"]
    except:
        extract_features_perpatient(data_path, pt_name, feature_param, outfolder, fn, df_suffix)
        data = np.load(os.path.join(outfolder, pt_name , f"data_{df_suffix}.npz"), allow_pickle=True)
        feature = data["feature"]
        starts = data["starts"]/feature_param["resample"]
        ends = data["ends"]/feature_param["resample"]
        channel_names = data["channel_names"]
    if len(feature) == 0:
        print(f"no data in {pt_name}")
        return
    batch_size = 32
    out = []
    for i in range(0, len(feature), batch_size):
        feature_batch = torch.tensor(feature[i:i+batch_size]).to(device)
        start_batch = starts[i:i+batch_size]
        end_batch = ends[i:i+batch_size]
        channel_name_batch = channel_names[i:i+batch_size]
        with torch.no_grad():
            feature_batch = preprocessing(feature_batch).float().to(device)
            output = model(feature_batch)
            output = output.cpu().numpy()
        for j in range(len(output)):
            out.append(pd.DataFrame({"start": [start_batch[j]], "end": [end_batch[j]], "channel_name": [channel_name_batch[j]], "preds": [output[j][0]]}))
    out_df = pd.concat(out, axis=0)
    out_df.to_csv(os.path.join(outfolder, pt_name, f"{df_suffix}_preds.csv"), index=False)        

def inference_pipeline(args, device, feature_param):
    model_path= args["model_fn"]
    ckpt = torch.load(model_path, map_location="cpu")
    model = ckpt["model"].to(device).float()
    model.eval()
    preprocessing_dict = ckpt["preprocessing"]
    preprocessing = PreProcessing.from_dict(preprocessing_dict)
    pt_names = os.listdir(args["data_dir"])
    for pt_name in sorted(pt_names):
        if os.path.isdir(os.path.join(args["data_dir"], pt_name)):
            print(pt_name)
            inference_one_patient(args["data_dir"], pt_name, model, device, feature_param, args["fn"], preprocessing)

if __name__ == "__main__":
    device = sys.argv[1]
    fn = sys.argv[2]
    args['device'] = device
    args["data_dir"] = "inference/zurich_bipolar"
    args["model_fn"] = "./result/artifact_data_win285_freq10_290_shift50/ckpt/fold_0/pruned_model.tar"
    args["fn"] = fn
    feature_param["n_jobs"] = 32
    feature_param["n_feature"] = 1         # 1 for time-frequency image, 2 for time-frequency image and amplitude coding plot
    feature_param["resample"] = 2000       # resample eeg signal 
    feature_param["time_window_ms"] = 1000  # time window for feature extraction
    feature_param["freq_min_hz"] = 10      # frequency min for time-frequency image
    feature_param["freq_max_hz"] = 500     # frequency max for time-frequency image
    feature_param["image_size"] = 224      # image size for feature extraction
    inference_pipeline(args, device, feature_param)