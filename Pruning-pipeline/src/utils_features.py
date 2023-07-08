import numpy as np
import  math
from scipy.interpolate import interp1d
import scipy.linalg as LA
import os 
import numpy as np 
from skimage.transform import resize
from multiprocessing import Process
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import mne
import pandas as pd
from HFODetector import ste, mni
import pandas as pd
import torch
from multiprocessing import Pool 
from multiprocessing import get_context
from p_tqdm import p_map

def generate_feature_from_df(df, data, channels_unique, sampling_rate, feature_param, n_jobs=8):
    '''
    df: pandas dataframe, with columns: start, end, channel_name
    '''
    starts = df["starts"].values.astype(int)
    ends = df["ends"].values.astype(int)
    channel_names = df["channel_names"].values
    hfo_waveforms = extract_waveforms(data, starts, ends, channel_names, channels_unique, int(feature_param["time_window_ms"]*2/1000*sampling_rate))
    sp_rate = [sampling_rate]*len(starts)
    win_size = [feature_param["image_size"]]*len(starts)
    ps_MinFreqHz = [feature_param["freq_min_hz"]]*len(starts)
    ps_MaxFreqHz = [feature_param["freq_max_hz"]]*len(starts)
    resize = [True]*len(starts)
    ret = p_map(hfo_feature, starts, ends, channel_names, hfo_waveforms, sp_rate, win_size, ps_MinFreqHz, ps_MaxFreqHz, resize, num_cpus=n_jobs)
    starts, ends, channel_names, time_frequncy_img, amplitude_coding_plot = np.zeros(len(ret)), np.zeros(len(ret)), np.empty(len(ret), dtype= object), np.zeros((len(ret), feature_param["image_size"], feature_param["image_size"])), np.zeros((len(ret), feature_param["image_size"], feature_param["image_size"]))
    for i in range(len(ret)):
        channel_names[i], starts[i], ends[i], time_frequncy_img[i], amplitude_coding_plot[i] = ret[i]
    return starts, ends, channel_names, time_frequncy_img, amplitude_coding_plot

def calcuate_boundary(start, end, length, win_len=2000):
    if start < win_len: 
        return 0, win_len
    if end > length - win_len: 
        return length - win_len, length
    return int(0.5*(start + end) - win_len//2), int(0.5*(start + end) + win_len//2)

def extract_waveforms(data, starts, ends, channel_names, unique_channel_names, window_len=2000):
    '''
    The extracted waveform will be (n_HFOs, widnow_len long) 
    
    '''

    def extract_data(data, start, end, window_len=2000):
        data = np.squeeze(data)
        start, end = calcuate_boundary(start, end, len(data), win_len=window_len)
        hfo_waveform = data[start:end]
        return hfo_waveform

    hfo_waveforms = np.zeros((len(starts), int(window_len)))
    for i in range(len(starts)):
        channel_name = channel_names[i]
        start = starts[i]
        end = ends[i]
        channel_index = np.where(unique_channel_names == channel_name)[0]
        hfo_waveform = extract_data(data[channel_index], start, end, window_len)
        hfo_waveforms[i] = hfo_waveform
    return hfo_waveforms
    
def hfo_feature(start, end, channel_name, data, sample_rate, win_size, ps_MinFreqHz, ps_MaxFreqHz, resize_img = True):
    # generate one sec time-freqeucny image
    spectrum_img = compute_spectrum(data, ps_SampleRate=sample_rate, ps_FreqSeg=win_size, ps_MinFreqHz=ps_MinFreqHz, ps_MaxFreqHz=ps_MaxFreqHz)
    amplitude_coding_plot = construct_coding(data, win_size) # height * len(selected_data)
    time_frequncy_img = spectrum_img
    if resize_img:
        time_frequncy_img = resize(time_frequncy_img, (win_size, win_size))
        amplitude_coding_plot = resize(amplitude_coding_plot, (win_size, win_size))
    return channel_name, start, end, time_frequncy_img, amplitude_coding_plot

def create_extended_sig(sig):
    s_len = len(sig)
    s_halflen = int(np.ceil(s_len/2)) + 1
    start_win = sig[:s_halflen] - sig[0]
    end_win = sig[s_len - s_halflen - 1:] - sig[-1]
    start_win = -start_win[::-1] + sig[0]
    end_win = -end_win[::-1] + sig[-1]
    final_sig = np.concatenate((start_win[:-1],sig, end_win[1:]))
    if len(final_sig)%2 == 0:
        final_sig = final_sig[:-1]
    return final_sig

def compute_spectrum(org_sig, ps_SampleRate = 2000, ps_FreqSeg = 512, ps_MinFreqHz = 10, ps_MaxFreqHz = 500, device = 'cpu'):
    device = torch.device(device)
    ii, jj = int(len(org_sig)//2), int(len(org_sig)//2 + len(org_sig))
    extend_sig = create_extended_sig(org_sig)
    extend_sig = torch.from_numpy(extend_sig).float()
    extend_sig = extend_sig.to(device)
    ps_StDevCycles = 3
    s_Len = len(extend_sig)
    s_HalfLen = math.floor(s_Len/2)+1
    v_WAxis = (torch.linspace(0, 2*np.pi, s_Len)[:-1]* ps_SampleRate).float()
    v_WAxisHalf = v_WAxis[:s_HalfLen].to(device).repeat(ps_FreqSeg, 1)
    v_FreqAxis = torch.linspace(ps_MaxFreqHz, ps_MinFreqHz,steps=ps_FreqSeg, device=device).float()
    v_WinFFT = torch.zeros(ps_FreqSeg, s_Len, device=device).float()
    s_StDevSec = (1 / v_FreqAxis) * ps_StDevCycles
    v_WinFFT[:, :s_HalfLen] = torch.exp(-0.5*torch.pow(v_WAxisHalf - (2 * torch.pi * v_FreqAxis.view(-1, 1)), 2) * (s_StDevSec**2).view(-1, 1))
    v_WinFFT = v_WinFFT * np.sqrt(s_Len)/ torch.norm(v_WinFFT, dim = -1).view(-1, 1)
    v_InputSignalFFT = torch.fft.fft(extend_sig)
    res = torch.fft.ifft(v_InputSignalFFT.view(1, -1)* v_WinFFT)[:, ii:jj]/torch.sqrt(s_StDevSec).view(-1,1)
    res = np.abs(res.numpy())
    return res


def construct_coding(raw_signal, height):
    index = np.arange(height)
    intensity_image = np.zeros((height, len(raw_signal)))
    intensity_image[index, :] = raw_signal
    return intensity_image


def clean_folder(saved_fn):
    if not os.path.exists(saved_fn):
        os.makedirs(saved_fn)
    else:
        shutil.rmtree(saved_fn)
        os.mkdir(saved_fn)

def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in enumerate(futures):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out