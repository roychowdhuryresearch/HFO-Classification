import numpy as np
import mne
import re

def ends_with_number(channel_name):
    # Regular expression to check if the channel name ends with -{number}
    return bool(re.search(r'-\d+$', channel_name))

def read_raw(raw_path, resample=2000, drop_duplicates=True):
    raw = mne.io.read_raw_edf(raw_path, verbose= False)
    if raw.info['sfreq'] != resample:
        raw = raw.resample(resample, npad='auto')
    raw_channels = raw.info['ch_names']
    data, channels = [], []

    for raw_ch in raw_channels:
        # if there is -{number} except -0, we skip it
        # check if -{number} is in the raw_ch
        # if ends_with_number(raw_ch) and "-0" not in raw_ch:
        #     continue
        ch_data = raw.get_data(raw_ch) * 1E6
        if drop_duplicates and "-0" in raw_ch:
            raw_ch = raw_ch.replace("-0", "")
        data.append(ch_data)
        channels.append(raw_ch)
    
    data = np.squeeze(np.array(data))
    return data, channels

def concate_edf(fns, resample):
    data, channel = [], []
    for fn in fns:
        d, c = read_raw(fn, resample = resample)
        data.append(d), channel.append(c)
    data = np.concatenate(data)
    channel = np.concatenate(channel)
    return data, channel

