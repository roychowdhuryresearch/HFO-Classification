from pyedflib import highlevel
import numpy as np
import os
import pickle
import scipy.io as sio
from scipy import signal
from sklearn.preprocessing import normalize
import shutil
import matplotlib.pylab as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score

import torch
### measurement: sensitivity, specificity, accuracy
def compute_metric(labels, predictions):
    
    specificity, recallTP = recall_score(labels, predictions, average=None)
    accuracy = accuracy_score(labels, predictions)

    res_dict = {"recall": recallTP, "specificity": specificity, "accuracy": accuracy}
    return res_dict

def clean_folder(saved_fn):
    if not os.path.exists(saved_fn):
        #os.mkdir(saved_fn)
        os.makedirs(saved_fn)
    else:
        shutil.rmtree(saved_fn)
        os.mkdir(saved_fn)

def read_edf(fn):
    print(fn)
    signals, signal_headers, header = highlevel.read_edf(fn)
    return signals, signal_headers, header

def parse_txt(fn):
    res = []
    with open(fn, "r") as f:
        lines = f.readlines()
    for l in lines:
        l = l.strip().split(",")
        temp = []
        for ll in l:
            temp.append(ll.strip())
        res.append(temp)
    return res

from PIL import Image

def save_imgs2dir(img_list, labels, name, output_dir):
    number_img = len(img_list)
    tmp_image_dir = os.path.join(output_dir, "tmp_imgs")
    if not os.path.exists(tmp_image_dir):
        os.makedirs(tmp_image_dir)
    for i in range(number_img):
        plt.figure(figsize=(10,10))
        sn.heatmap(img_list[i],cmap="viridis", annot = labels, vmin=0, vmax=1,fmt = '')
        plt.savefig(os.path.join(tmp_image_dir, f'{name}_{i}.jpg'))
    print(f"generated images to {tmp_image_dir}")

def dict2array(dic):
    res = []
    print(sorted(dic.keys()))
    for key in sorted(dic.keys()):
        res.append(dic[key])
    return np.concatenate(res)

def dump_pickle(saved_fn, variable):
    with open(saved_fn, 'wb') as ff: 
        pickle.dump(variable, ff)

def load_pickle(fn):
    if not os.path.exists(fn):
        print(fn , " notexist")
        return
    with open(fn, "rb") as f:
        lookup = pickle.load(f)
        #print(fn)
    return lookup

##for each channel, specify the width of the image
def construct_win_image(data, num_win, stride_size, mode="log_norm"):
    num_index = data.shape[1]
    new_segments = [] 
    if mode == "log_norm":
        data = np.log(data + 1e-8)
    for win_idx in range(0, num_index - num_win, stride_size):
        new_segments.append(data[:, win_idx: (win_idx + num_win)])
    return new_segments

def construct_window_seg(fn, patient_name):
    loaded_data = load_pickle(fn)
    all_segments = {}
    #channel_name
    num_win = 10
    stride_size = 5
    keys = list(loaded_data.keys())
    for key in keys:
        new_segments = construct_win_image(loaded_data[key], num_win, stride_size)
        all_segments[key] = new_segments
    return all_segments


def standardize_length(signals, length, shift_mean=False, magnitute_normalize=False):
    signal_standard = np.zeros((len(signals), length))
    for cnt, sig in enumerate(signals):
        signal_resample = signal.resample(sig, length)
        if shift_mean:
            signal_resample = signal_resample - np.mean(signal_resample)
       
        signal_standard[cnt, :] = signal_resample
    if magnitute_normalize:
        signal_standard = normalize(signal_standard, norm='l2')
    return signal_standard

def plot_heatmap(data, hori_label, verti_label, title, loc):
    # Create a dataset (fake)
    plt.close("all")
    plt.figure(figsize=(20,20))
    df = pd.DataFrame(data, index= verti_label, columns=hori_label)
    # Default heatmap: just a visualization of this square matrix
    p1 = sn.heatmap(df, cmap=plt.cm.Blues, annot=True )
    plt.title(title)
    fn = os.path.join(loc, title+".jpg")
    plt.savefig(fn)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def expand_dim(value, dim):
    if len(value.shape) != dim:
        value = value.unsqueeze(0)
    return value

def normalize_img(a):
    batch_num = a.shape[0]
    h = a.shape[1]
    w = a.shape[2]
    a_reshape = a.reshape(batch_num, -1)
    a_min = torch.min(a_reshape, -1)[0].unsqueeze(1)
    a_max = torch.max(a_reshape, -1)[0].unsqueeze(1)
    c = 255.0 * (a_reshape - a_min)/(a_max - a_min)
    c = c.reshape(batch_num,h, w)
    return c
