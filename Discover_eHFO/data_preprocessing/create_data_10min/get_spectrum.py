
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from scipy.io import loadmat
import numpy as np 
from tqdm import tqdm
import numpy as np
from utils import *
from ConfigParse import *

def expand_dims(x, num):
    if len(x.shape) == num:
        x = np.expand_dims(x, axis=0)
    return x

def channel_features(channel_data, key):
    data = np.array(channel_data.v_Intervals)
    info = np.array(channel_data.st_HFOInfo.m_EvtLims)
    spectrum, waveform, intensity = [], [], []
    if not hasattr(data, "__len__"):
        print(data, key)
    if len(data) == 0:
        return None, None, None, None, None

    if not hasattr(data[0], "__len__"):
        data = data.reshape(1, -1)
    for i in range(len(data)):
        spectrum_img = compute_spectrum(data[i])
        spike_image, intensity_image = construct_features(data[i][1000:3000])
        spectrum.append(spectrum_img)
        waveform.append(spike_image)
        intensity.append(intensity_image)
    info_list = expand_dims(np.array(info), 1)
    
    channel_name = [strip_key(key)]*len(data)
    spectrum = expand_dims(np.array(spectrum), 2)
    waveform = expand_dims(np.array(waveform),2)
    intensity = expand_dims(np.array(intensity),2)
    return channel_name, info_list, spectrum, waveform, intensity


def genenerate_spectrum(in_folder, patient_name, out_folder):
    out_folder = os.path.join(out_folder, patient_name)
    in_folder = os.path.join(in_folder, patient_name)
    #in_folder = os.path.join(in_folder, patient_name, "HFO events")
    all_fn = os.listdir(in_folder)
    valid_fn = []
    for fn in all_fn:
        if fn.endswith(".rhfe") and "original" in fn:
            valid_fn.append(os.path.join(in_folder, fn))
    channel_name, interval ,spectrum, waveform, intensity = [], [], [], [], []
    for fn in valid_fn:
        x = loadmat(fn, squeeze_me=True,struct_as_record=False)
        real_key = [item for item in x.keys() if item not in set(['__header__', '__version__', '__globals__', 'st_FileData'])]
        param_list = [{"channel_data":x[k], "key":k} for k in real_key]
        ret = parallel_process(param_list, channel_features, n_jobs=16, use_kwargs=True, front_num=3)
        for j in tqdm(ret):
            if not type(j) is tuple:
                print(j)
            if j[0] is None:
                continue
            channel_name.append(j[0])
            interval.append(j[1])
            spectrum.append(j[2])
            waveform.append(j[3])
            intensity.append(j[4])
    channel_name = np.squeeze(np.concatenate(channel_name,0))
    interval = np.squeeze(np.concatenate(interval,0))
    spectrum = np.squeeze(np.concatenate(spectrum,0))
    waveform = np.squeeze(np.concatenate(waveform,0))
    intensity = np.squeeze(np.concatenate(intensity,0))
    os.mkdir(out_folder)
    np.savez_compressed(os.path.join(out_folder,"data.npz"), info=channel_name, start_end=interval, spectrum = spectrum, waveform= waveform, intensity=intensity)


if __name__ == "__main__":
    args = arg_parse(sys.argv[1:])
    print("source folder: ", args.datain_10)
    print("res_folder", args.dataout_10)
    in_folder = args.datain_10
    out_folder = args.dataout_10
    clean_folder(out_folder)
    for patient_name in os.listdir(in_folder):
        print(patient_name)
        if patient_name == "NS_long":
            continue
        genenerate_spectrum(in_folder, patient_name, out_folder)

