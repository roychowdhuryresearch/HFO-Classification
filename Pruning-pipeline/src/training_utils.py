import torch
import numpy as np
import shutil
import os, copy

def pack_batch(pt_names, data, label, channel_name, start_end, device):
    channel_name = np.array(channel_name)
    pt_name = np.array(pt_names)
    data_norm = normalize_img(data.float()).to(device)
    label = label.squeeze().float().to(device)
    batch = {
        "inputs": data_norm,
        "label": label, 
        "channel_name":channel_name, 
        "start_end":start_end,
        "pt_name":pt_name
    }
    return batch

def clean_folder(saved_fn):
    if not os.path.exists(saved_fn):
        #os.mkdir(saved_fn)
        os.makedirs(saved_fn)
    else:
        shutil.rmtree(saved_fn)
        os.mkdir(saved_fn)

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
    c = a.shape[1]
    h = a.shape[2]
    w = a.shape[3]
    a_reshape = a.reshape(batch_num * c, -1)
    a_min = torch.min(a_reshape, -1)[0].unsqueeze(1)
    a_max = torch.max(a_reshape, -1)[0].unsqueeze(1)
    normalized = 255.0 * (a_reshape - a_min)/(a_max - a_min)
    normalized = normalized.reshape(batch_num,c, h, w)
    return normalized

def save_checkpoint(state, filename='models/checkpoint.pth.tar'):
    torch.save(state, filename)

def pick_best_model(model, best_model ,epoch, v_loss, best_loss, checkpoint_folder, model_name="a", preprocessing=None, save = True, verbose = True):
    if v_loss < best_loss:
        best_loss = v_loss 
        best_model = copy.deepcopy(model)
        pre_processing = preprocessing.to_dict()
        if verbose:
            print(f"best_model of {model_name} in epoch ", epoch +1)
        if save:
            save_checkpoint({'epoch': epoch + 1,
                'model': best_model,
                "model_param":{"in_channels": model.in_channels, "outputs": model.outputs,"model_name": model_name},
                "preprocessing": pre_processing,
                },
                filename= os.path.join(checkpoint_folder, f'model_{model_name}.tar'))
    return best_loss, best_model
