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

def compute_tf_fig(org_sig):
    final_sig = create_extended_sig(org_sig)
    wave2000 = final_sig
    ps_SampleRate = 2000
    s_Len = len(final_sig)
    #exts_len = len(final_sig)
    s_HalfLen = math.floor(s_Len/2)+1
    ps_MinFreqHz = 10
    ps_MaxFreqHz = 500
    ps_FreqSeg = 512

    v_WAxis = np.linspace(0, 2*np.pi, s_Len, endpoint=False)
    v_WAxis = v_WAxis* ps_SampleRate
    v_WAxisHalf = v_WAxis[:s_HalfLen]
    v_FreqAxis = np.linspace(ps_MinFreqHz, ps_MaxFreqHz,num=ps_FreqSeg)#ps_MinFreqHz:s_FreqStep:ps_MaxFreqHz
    v_FreqAxis = v_FreqAxis[::-1]
    
    v_InputSignalFFT = np.fft.fft(wave2000)
    ps_StDevCycles = 3
    m_GaborWT = np.zeros((ps_FreqSeg, s_Len),dtype=complex)
    for i, s_FreqCounter in enumerate(v_FreqAxis):
        v_WinFFT = np.zeros(s_Len)
        s_StDevSec = (1 / s_FreqCounter) * ps_StDevCycles
        v_WinFFT[:s_HalfLen] = np.exp(-0.5*np.power( v_WAxisHalf - (2* np.pi* s_FreqCounter) , 2)*
            (s_StDevSec**2))
        v_WinFFT = v_WinFFT* np.sqrt(s_Len)/ LA.norm(v_WinFFT, 2)
        m_GaborWT[i, :] = np.fft.ifft(v_InputSignalFFT* v_WinFFT)/np.sqrt(s_StDevSec)
    return s_HalfLen, v_FreqAxis, v_WAxisHalf, v_InputSignalFFT,  m_GaborWT


def compute_spectrum(org_sig):
    final_sig = create_extended_sig(org_sig)
    wave2000 = final_sig
    ps_SampleRate = 2000
    s_Len = len(final_sig)
    
    #exts_len = len(final_sig)
    s_HalfLen = math.floor(s_Len/2)+1
    ps_MinFreqHz = 10
    ps_MaxFreqHz = 500
    ps_FreqSeg = 512

    v_WAxis = np.linspace(0, 2*np.pi, s_Len, endpoint=False)
    v_WAxis = v_WAxis* ps_SampleRate
    v_WAxisHalf = v_WAxis[:s_HalfLen]
    v_FreqAxis = np.linspace(ps_MinFreqHz, ps_MaxFreqHz,num=ps_FreqSeg)#ps_MinFreqHz:s_FreqStep:ps_MaxFreqHz
    v_FreqAxis = v_FreqAxis[::-1]
    
    v_InputSignalFFT = np.fft.fft(wave2000)
    ps_StDevCycles = 3
    m_GaborWT = np.zeros((ps_FreqSeg, s_Len),dtype=complex)
    for i, s_FreqCounter in enumerate(v_FreqAxis):
        v_WinFFT = np.zeros(s_Len)
        s_StDevSec = (1 / s_FreqCounter) * ps_StDevCycles
        v_WinFFT[:s_HalfLen] = np.exp(-0.5*np.power( v_WAxisHalf - (2* np.pi* s_FreqCounter) , 2)*
            (s_StDevSec**2))
        v_WinFFT = v_WinFFT* np.sqrt(s_Len)/ LA.norm(v_WinFFT, 2)
        m_GaborWT[i, :] = np.fft.ifft(v_InputSignalFFT* v_WinFFT)/np.sqrt(s_StDevSec)
    return resize(np.abs(m_GaborWT[:, 3000:5000]), (224,224))

def create_extended_sig(wave2000):
    #wave2000 = bb
    s_len = len(wave2000)
    s_halflen = int(np.ceil(s_len/2)) + 1
    sig = wave2000
    start_win = sig[:s_halflen] - sig[0]
    end_win = sig[s_len - s_halflen - 1:] - sig[-1]
    start_win = -start_win[::-1] + sig[0]
    end_win = -end_win[::-1] + sig[-1]
    final_sig = np.concatenate((start_win[:-1],sig, end_win[1:]))
    #print(s_halflen, start_win.shape, end_win.shape, sig.shape, final_sig.shape)
    if len(final_sig)%2 == 0:
        final_sig = final_sig[:-1]
    return final_sig


def strip_key(key):
    key = key.strip()
    key = key.replace('EEG', '').strip()
    key = key.replace('Ref', '').strip()
    key = key.replace('-', '').strip()
    key = key.replace('_', ' ').strip()
    key = key.split(" ")
    if len(key) > 1:
        key = key[1]
    else:
        key = key[0]     
    return key

def normalized(a, max_ = 2000-11):
    c = (max_*(a - np.min(a))/np.ptp(a)).astype(int)
    c = c + 5 
    return c 

def construct_features(raw_signal, length=1000):
    #HFO with spike
    canvas = np.zeros((2*length, 2*length))
    hfo_spike = normalized(raw_signal)
    index = np.arange(len(hfo_spike))
    for ii in range(3):
        canvas[index,hfo_spike-ii] = 256
        canvas[index,hfo_spike+ii] = 256 
    spike_image = resize(canvas, (224, 224))

    intensity_image = np.zeros_like(canvas)
    intensity_image[index, :] = raw_signal
    hfo_image = resize(intensity_image, (224, 224))

    return spike_image, hfo_image

def clean_folder(saved_fn):
    if not os.path.exists(saved_fn):
        #os.mkdir(saved_fn)
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
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out