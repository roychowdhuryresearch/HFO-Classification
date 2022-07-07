## merge bank HFO results from matlab HFO detectors for each patient
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
pparentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
sys.path.insert(0,pparentdir) 
import numpy as np
import scipy.io as sio
from scipy.io import loadmat
import numpy as np
import pandas as pd
from scipy.io.matlab.mio4 import arr_to_2d 

from utils import *
from ConfigParse import *
from patient_info import seizure_free_patient_names


def parse_artifacts_prediction(annotation, csv_dir ,pt_name):
    """
    parse_artifacts_prediction from 90 min inference from model trained in 10 min 
    """
    df = pd.DataFrame()   
    df['channel_name'] = np.char.upper(np.squeeze(annotation["info"]))
    start_end =  np.squeeze(annotation["start_end"])
    df['start'] = start_end[:, 0]
    df['end'] = start_end[:, 1]
    fn = f"{pt_name}.csv"
    hfo_df = pd.read_csv(os.path.join(csv_dir, fn))
    hfo_df['channel_names'] = np.char.upper(np.array(list(hfo_df['channel_names'].values)))
    new_df = pd.merge(df, hfo_df,  how='left', left_on=['channel_name','start', 'end'], right_on = ['channel_names','start', 'end'])
    artifacts_label = new_df['predictions_a'].values
    return artifacts_label

def parse_patient_label(df,channel_names, mode):
    """
    Get soz and removed information from csv
    """
    labels = np.zeros(len(channel_names))
    df["Ch"] = np.char.upper(df["Ch"].values.astype(str))
    channel_names = np.char.upper(channel_names)
    channel_unique = np.unique(df['Ch'].values.astype(str))
    for c in channel_unique:
        indices = np.where(channel_names == np.char.upper(c))[0]
        channel_label = df.loc[df['Ch'] == np.char.upper(c)][mode].to_numpy()[0]
        labels[indices] = channel_label
    return labels

def get_remove_label(input_dir, annotation_fn, inference_res90min):
    col = ["Ch", "SOZ", "Removed"]
    dfs = pd.read_excel(annotation_fn, sheet_name=None, usecols = col)

    for pn in os.listdir(input_dir):
        df_p = dfs[pn]
        patient_folder = os.path.join(input_dir, pn) 
        loaded = np.load(os.path.join(patient_folder,"data.npz"), allow_pickle=True)
        channel_names = loaded["info"]
        remove_labels = parse_patient_label(df_p,np.squeeze(channel_names), "Removed")
        soz_labels = parse_patient_label(df_p,np.squeeze(channel_names), "SOZ")
        artifacts_prediction = parse_artifacts_prediction(loaded, inference_res90min ,pn)


        np.savez(os.path.join(patient_folder,"label.npz"),\
                              remove_labels=remove_labels,\
                              soz_labels= soz_labels, artifacts = artifacts_prediction)
    
    #non_seizure_free_patient = list(set(os.listdir(input_dir)) - set(seizure_free_patient_names))
    #for pn in seizure_free_patient_names:

if __name__ == "__main__":
    args = arg_parse(sys.argv[1:])
    print("res_folder", args.dataout_90)
    get_remove_label(args.dataout_90, args.channel_annotation, args.inference_res)