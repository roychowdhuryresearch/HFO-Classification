from utilities import dump_pickle
import numpy as np
import os
import pandas as pd
import seaborn as sns
import numpy as np
def fetch_hfo_add_info(stats, csv_dir ,pt_name):
    """
    Get soz and removed, spike information from csv
    """
    df = pd.DataFrame()   
    df['channel_name'] = np.char.upper(np.squeeze(stats["info"]))
    start_end =  np.squeeze(stats["start_end"])
    df['start'] = start_end[:, 0]
    df['end'] = start_end[:, 1]
    fn = f"{pt_name}.csv"
    hfo_df = pd.read_csv(os.path.join(csv_dir, fn))
    hfo_df['channel_names'] = np.char.upper(np.array(list(hfo_df['channel_names'].values)))
    new_df = pd.merge(df, hfo_df,  how='left', left_on=['channel_name','start', 'end'], right_on = ['channel_names','start', 'end'])
    dff = new_df[["channel_name",'start', 'end', 'predictions_a']]
    artifacts_label = new_df['predictions_a'].values
    return artifacts_label
    #print(new_df.head)

def add_prediction(in_folder, csv_dir):
    valid_patients = os.listdir("/media/yipeng/data/HFO_clasification/HFO_wise_stats_90")
    valid = set()
    for pp in valid_patients:
        valid.add(pp.split(".")[0])
    for pn in os.listdir(in_folder):
        if pn not in valid:
            continue
        folder = os.path.join(in_folder, pn)
        loaded = np.load(os.path.join(folder,"data_flzoomin.npz"), allow_pickle=True) 
        artifacts_label = fetch_hfo_add_info(loaded, csv_dir ,pn)
        dump_pickle(os.path.join(folder,"artifacts.pkl"),artifacts_label)

if __name__ == "__main__":
    in_folder = "/media/yipeng/data/HFO_clasification/HFO_classification_training_data_spike_90_10_500"
    add_prediction(in_folder)