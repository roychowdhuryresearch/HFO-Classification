# what we are expecting is one folder per patient, and inside each folder there several edf files but one csv file.
import os
from src.utils_edf import concate_edf
from src.utils_features import generate_feature_from_df
import pandas as pd
import numpy as np
from src.args import data_args as feature_param
from src.training_utils import clean_folder

def extract_features_perpatient(folder, patient_names, feature_param, outfolder, fn = 'annotation.csv', suffix = ''):
    patient_folder = os.path.join(folder, patient_names)
    edf_files = [os.path.join(patient_folder, f) for f in os.listdir(patient_folder) if f.endswith('.edf')]
    csv_files = os.path.join(patient_folder, fn)
    annotation_df = pd.read_csv(csv_files)
    data, channels = concate_edf(edf_files, resample=feature_param["resample"])
    # convert time to sample
    annotation_df["starts"] = (annotation_df["starts"]*feature_param["resample"]).astype(int)
    annotation_df["ends"] = (annotation_df["ends"]*feature_param["resample"]).astype(int)
    starts, ends, channel_names, time_frequncy_img, amplitude_coding_plot = generate_feature_from_df(annotation_df, data, channels, sampling_rate=feature_param["resample"], feature_param=feature_param, n_jobs=feature_param["n_jobs"])
    if feature_param["n_feature"] == 1:
        feature = time_frequncy_img[:, None, :, :]
    else:
        feature = np.concatenate((time_frequncy_img[:, None, :, :], amplitude_coding_plot[:, None, :, :]), axis=1)
    outfolder = os.path.join(outfolder, patient_names)
    os.makedirs(outfolder, exist_ok=True)
    if suffix != '':
        suffix = f"data_{suffix}.npz" 
    else:
        suffix = "data.npz" 
    np.savez_compressed(os.path.join(outfolder, suffix), feature=feature, starts=starts, ends=ends, channel_names=channel_names, labels = annotation_df['labels'].values)

def extract_features(folder, feature_param, outfolder):
    patient_names = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    for patient_name in patient_names:
        extract_features_perpatient(folder, patient_name, feature_param, outfolder)
    # make feature param to csv
    feature_param_df = pd.DataFrame(feature_param, index=[0])
    feature_param_df.to_csv(os.path.join(outfolder, "data_meta.csv"), index=False)

if __name__ == "__main__":
    data_folder = "data/artifact_data"
    outfolder = f"data_training/{data_folder.split('/')[-1]}"
    feature_param["n_jobs"] = 8
    feature_param["n_feature"] = 1         # 1 for time-frequency image, 2 for time-frequency image and amplitude coding plot
    feature_param["resample"] = 2000       # resample eeg signal 
    feature_param["time_window_ms"] = 1000  # time window for feature extraction
    feature_param["freq_min_hz"] = 10      # frequency min for time-frequency image
    feature_param["freq_max_hz"] = 500     # frequency max for time-frequency image
    feature_param["image_size"] = 224      # image size for feature extraction
    clean_folder(outfolder)
    extract_features(data_folder, feature_param, outfolder)
