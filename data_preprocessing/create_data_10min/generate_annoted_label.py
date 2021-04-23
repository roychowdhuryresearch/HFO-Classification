## merge bank HFO results from matlab HFO detectors for each patient
import numpy as np
import scipy.io as sio
from scipy.io import loadmat
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
from scipy.io.matlab.mio4 import arr_to_2d 
sys.path.insert(0,parentdir)
from utils import *
from ConfigParse import *


def get_patient_annotation(folder, suffix):
    res = {}
    invalid_keys = set(['__header__', '__version__', '__globals__', 'st_FileData'])
    for p_name in os.listdir(folder):
        
        p_folder = os.path.join(folder, p_name)
        valid_files = []
        for fn in os.listdir(p_folder):
            if fn.endswith(suffix):
                valid_files.append(os.path.join(p_folder,fn))      
        p_res = set()
        for fn in valid_files:
            loaded = loadmat(fn, squeeze_me=True,struct_as_record=False)
            for channel_name in loaded.keys():
                if channel_name in invalid_keys:
                    continue
                channel_name_parsed = strip_key(channel_name)
                channel_intervals = np.array(loaded[channel_name].st_HFOInfo.m_EvtLims).astype(int)
                if len(channel_intervals.shape) == 1:
                    channel_intervals = np.expand_dims(channel_intervals, 0)
                for start_end in channel_intervals:
                    if len(start_end) == 0:
                        continue
                    p_res.add("#".join([str(item) for item in [channel_name_parsed]+ list(start_end)]))
                    #p_res.add(str(channel_name_parsed) + "#" + "")
        res[p_name] = p_res
    return res

def create_patient_label(channel_names, start_end,real_hfo, spike_hfo):
    label = np.zeros(len(channel_names))
    for idx, c in enumerate(channel_names):
        s_e = start_end[idx]
        key = "#".join([str(item) for item in [c]+ list(s_e)])
        label[idx] = int(key in real_hfo) + int(key in spike_hfo)
    return label

def create_labels(spectrum_folder, original_data_folder):
    artifacts_stats = get_patient_annotation(original_data_folder, "verified_STE.rhfe")
    spike_stats = get_patient_annotation(original_data_folder, "HFO-spike_STE.rhfe")
    for p_name in os.listdir(spectrum_folder):
        p_loded = np.load(os.path.join(spectrum_folder, p_name, "data.npz")) # 3 image files
        p_channel_names = p_loded["info"]
        p_start_end = p_loded["start_end"]
        p_label = create_patient_label(p_channel_names, p_start_end, artifacts_stats[p_name], spike_stats[p_name])
        np.savez(os.path.join(spectrum_folder, p_name, "label.npz"), labels= p_label)

if __name__ == "__main__":
    args = arg_parse(sys.argv[1:])
    print("res folder: ", args.dataout_10)
    print("source folder", args.datain_10)
    spectrum_folder = args.dataout_10
    original_data_folder = args.datain_10
    create_labels(spectrum_folder, original_data_folder)