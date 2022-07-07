import os
from src.utilities import *
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

def generate_stats_df(folder, name="spikes.pkl"):
    stats_fn = os.path.join(folder, name)
    stats = load_pickle(stats_fn)
    outputs = stats["outputs"]
    channel_names = stats["channel_name"].astype(str)
    start_end = stats["start_end"]
    labels=stats["labels"]
  
    df = pd.DataFrame()   
    df['channel_name'] = np.char.upper(np.squeeze(channel_names))
    df['outputs'] = np.squeeze(outputs)
    df['start'] = np.squeeze(start_end)[:,0]
    df['end'] = np.squeeze(start_end)[:,1]
    df['labels'] = np.squeeze(labels)
    df['predictions'] = df['outputs'].values > 0.5

    return df

def get_performance(df):
    prediction = df['predictions'].values.astype(int)
    labels = (df['labels'].values).astype(int)
    precision , recall, f_1, _ = precision_recall_fscore_support(labels, prediction)
    acc = accuracy_score(labels, prediction)
    recall = recall[1] 
    precision = precision[1]
    f1 = f_1[1]
    return acc, recall, precision, f1

def save_patch(v, channel_name, save_dir):
    fig, ax = plt.subplots(figsize=(0.5, 0.5))
    fig.subplots_adjust(0,0,1,1)
    plt.gca().set_axis_off()
    plt.margins(0,0)
    x = np.zeros((3,3))+v
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    color_map = plt.imshow(x)
    #color_map.set_cmap("Blues_r")
    plt.clim(0,1)
    ax.set_yticklabels([],  fontsize=0)
    ax.set_xticklabels([],  fontsize=0)
    plt.savefig(os.path.join(save_dir,channel_name+".png"),bbox_inches = 'tight',pad_inches = 0)
    plt.close()

def save_patch(v, channel_name, save_dir):
    fig, ax = plt.subplots(figsize=(0.5, 0.5))
    fig.subplots_adjust(0,0,1,1)
    plt.gca().set_axis_off()
    plt.margins(0,0)
    cmap = plt.cm.get_cmap('viridis')
    circle1=plt.Circle((0,0),.03,color=cmap(v))
    plt.gcf().gca().add_artist(circle1)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #color_map = plt.imshow(x)
    #color_map.set_cmap("Blues_r")
    #plt.clim(0,1)
    ax.set_yticklabels([],  fontsize=0)
    ax.set_xticklabels([],  fontsize=0)
    plt.savefig(os.path.join(save_dir,channel_name+".png"),bbox_inches = 'tight',pad_inches = 0,transparent=True)
    plt.close()


def normalize(x):
    return (x-min(x))/(max(x)-min(x))

def generate_patches(c_name, predictions ,save_dir): 
    #predictions = np.log(predictions+1).tolist()
    predictions = predictions.tolist()
    predictions.append(0)
    predictions = np.array(predictions)
    normalized = normalize(predictions)
    c_name = c_name.tolist()
    c_name.append("Zero")
    for idx, cn in enumerate(c_name):
        save_patch(normalized[idx], cn, save_dir)


def hist_patients_pathological_count(in_fn, out_folder,patch=False):
    for p in os.listdir(in_fn):
        p_folder = os.path.join(in_fn, p)
        p_df = generate_stats_df(p_folder, name="spikes.pkl")
        p_df_agg = p_df.groupby('channel_name').agg({'predictions': 'sum'}).reset_index()
        fig =plt.figure(figsize=(25, 5))
        ax = fig.add_subplot(111)
        c_name = p_df_agg['channel_name']
        predictions = p_df_agg['predictions']
        ax.bar(c_name,predictions)
        fig.savefig(os.path.join(out_folder, p+".jpg"))
        plt.close(fig)
        if patch:
            save_dir = os.path.join(out_folder, p)
            clean_folder(save_dir)
            generate_patches(c_name, predictions ,save_dir)



def hist_patients_physilogical_count(in_fn, out_folder,patch=False):
    for p in os.listdir(in_fn):
        p_folder = os.path.join(in_fn, p)
        p_df = generate_stats_df(p_folder, name="spikes.pkl")
        p_df["predictions"] = (p_df["predictions"].values == 0).astype(int)
        p_df_agg = p_df.groupby('channel_name').agg({'predictions': 'sum'}).reset_index()
        fig =plt.figure(figsize=(25, 5))
        ax = fig.add_subplot(111)
        c_name = p_df_agg['channel_name']
        predictions = p_df_agg['predictions'].values
        ax.bar(c_name,predictions)
        fig.savefig(os.path.join(out_folder, p+".jpg"))
        plt.close(fig)
        if patch:
            save_dir = os.path.join(out_folder, p)
            clean_folder(save_dir)
            generate_patches(c_name, predictions ,save_dir)


def csv_kfold_performance(in_fn, out_folder, mode = "spikes"):
    df = pd.DataFrame(columns=['fold', "acc", "recall","precision", "f1"])
    for p in os.listdir(in_fn):
        p_folder = os.path.join(in_fn, p)
        p_df = generate_stats_df(p_folder, name= mode+".pkl")
        acc, recall, precision, f1 = get_performance(p_df)
        df = df.append({"fold":p,"acc":acc, "recall":recall, "precision": precision, "f1":f1}, ignore_index=True)
    df = df.append({"fold":"mean","acc":np.mean(df["acc"].values), "recall":np.mean(df["recall"].values), "precision": np.mean(df["precision"].values), "f1":np.mean(df["f1"].values)}, ignore_index=True)
    df.to_csv(os.path.join(out_folder, mode+".csv"))