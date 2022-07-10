import numpy as np
import os
import pandas as pd
from src.utilities import load_pickle

def process_pkl_file(fn, sz_is_1 = True):
    stats = load_pickle(os.path.join(fn, "spikes.pkl"))
    df = pd.DataFrame()
    if len(stats["start_end"]) == 0:
        return None
    df["start"] = np.squeeze(stats["start_end"])[:,0]
    df["end"] = np.squeeze(stats["start_end"])[:,1]
    #df["num_behavior"] = (np.squeeze(stats["outputs"]) > 0.5).astype(int)
    if sz_is_1:
        df["num_sz_ad"] = (np.squeeze(stats["outputs"]) > 0.5).astype(int)
        df["num_behavior"] = (np.squeeze(stats["outputs"]) < 0.5).astype(int)
    else:
        df["num_sz_ad"] = (np.squeeze(stats["outputs"]) < 0.5).astype(int)
        df["num_behavior"] = (np.squeeze(stats["outputs"]) > 0.5).astype(int)
    df["channel_names"] = np.squeeze(stats["channel_name"])
    return df

def process_res_csv_file(fn, sz_is_1 = True, sti_only=False):
    if sti_only:
        df = process_csv_file(os.path.join(fn, "inference_new.csv"))
    else:
        df = process_csv_file(os.path.join(fn, "inference.csv"))
    #df["num_behavior"] = (np.squeeze(stats["outputs"]) > 0.5).astype(int)
    if sz_is_1:
        df["num_sz_ad"] = (df["outputs"] > 0.5).astype(int)
        df["num_behavior"] = (df["outputs"] < 0.5).astype(int)
    else:
        df["num_sz_ad"] = (df["outputs"] < 0.5).astype(int)
        df["num_behavior"] = (df["outputs"] > 0.5).astype(int)
    df["channel_names"] = df["channel_name"]
    return df


def process_csv_file(fn):
    df = pd.read_csv(fn)
    return df

def process_stimulate_label(fn, filtered_labels=False, relabeled=False):
    df = pd.DataFrame()
    if filtered_labels:
        loaded = np.load(os.path.join(fn, "label_stimulation_filtered.npz"))
        loaded_data = np.load(os.path.join(fn, "data_filtered.npz"))    
    elif relabeled:
        loaded = np.load(os.path.join(fn, "label_stimulation_relabeled.npz"))
        loaded_data = np.load(os.path.join(fn, "data_relabeled.npz"))
    else:
        loaded = np.load(os.path.join(fn, "label_stimulation.npz"))
        loaded_data = np.load(os.path.join(fn, "data.npz"))
        
    artifacts_prediction = loaded["artifacts"]
    channel_names = np.squeeze(loaded_data["info"])
    start_end = np.squeeze(loaded_data["start_end"])
    stimulate_labels = np.squeeze(loaded["stimulate_labels"])
    behavior_labels = np.squeeze(loaded["behavior_labels"])
    bad_labels = np.squeeze(loaded["bad_labels"])
    df["channel_names"] = channel_names
    df["start"] = start_end[:,0]
    df["end"] = start_end[:,1]
    df["stimulated"] = stimulate_labels
    df["artifacts"] = artifacts_prediction
    df["behavior"] = behavior_labels
    df["bad"] = bad_labels
    df["both"] = np.logical_and(behavior_labels, bad_labels).astype(int)
    df["bad_only"] = np.logical_xor(behavior_labels, bad_labels) * bad_labels
    df["behavior_only"] = np.logical_xor(behavior_labels, bad_labels) * behavior_labels
    df["none"] = np.logical_not(np.logical_or(behavior_labels, bad_labels)).astype(int)
    return df

def process_annotation_label(fn):
    df = pd.DataFrame()
    loaded = np.load(os.path.join(fn, "label.npz"))
    loaded_data = np.load(os.path.join(fn, "data.npz"))
    channel_names = np.squeeze(loaded_data["info"])

    for (i, ch) in enumerate(channel_names):
        if "Dep" in ch:
            channel_names[i] = ch[:2] + ch[-1]

    start_end = np.squeeze(loaded_data["start_end"])
    df["annot_soz"] = np.squeeze(loaded["soz_labels"])
    df["annot_remove"] = np.squeeze(loaded["remove_labels"])
    # df["spike"] = np.squeeze(loaded["spike"]) 
    df["channel_names"] = channel_names
    df["start"] = start_end[:,0]
    df["end"] = start_end[:,1]
    return df

    
    
