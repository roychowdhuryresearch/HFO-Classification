#!/usr/bin/env python
# coding: utf-8
## Two separate models for artifact classification and HFO with spike classification
### 

import os, sys, time, copy
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from random import random
import random
import numpy as np

from src.utilities import *
from src.dataloader import create_patient_eliminate_loader, create_kfold_loader
from src.model import NeuralCNN
from src.config import arg_parse
from src.meter import TrainingMeter, Meter
from src.training_utils import *

def validate(val_loader, model, criterion, computing_device, fn = None):
    start = time.time()
    meter_s = Meter("spike")
    meter_a = Meter("artifact")
    model_s = model["spike"]
    model_a = model["artifact"]
    model_a.eval()
    model_s.eval()
    for _, (spectrum, waveform, intensity, label, info, start_end) in enumerate(val_loader, 0):
  
        with torch.no_grad():       
            a_, s_ , train_s = create_sa_labels(spectrum, waveform, intensity,label, info, start_end, computing_device)
            outputs_a = model_a(a_["inputs"]).squeeze()
            
            a_["label"] = a_["label"].squeeze()

            if outputs_a.dim() == 0:
                outputs_a = outputs_a.unsqueeze(0)
                a_["label"] = a_["label"].unsqueeze(0)
                
            loss_a = criterion(outputs_a, a_["label"])
            
            if not fn:
                meter_a.update_loss(loss_a.detach().cpu().numpy())
                meter_a.update_outputs(outputs_a.detach().cpu().numpy(), a_["label"].cpu().numpy())
            else:
                meter_a.add(a_["spectrum"],
                        a_["label"].detach().cpu().numpy(),
                        a_["channel_name"],
                        a_["start_end"],
                        a_["intensity"], 
                        a_["waveform"],
                        outputs_a.detach().cpu().numpy())
            ## second
            if not train_s:
                continue
            outputs_s = model_s(s_["inputs"]).squeeze()
            s_["label"] = s_["label"].squeeze()
            
            if outputs_s.dim() == 0:
                outputs_s = outputs_s.unsqueeze(0)
                s_["label"] = s_["label"].unsqueeze(0)               
            loss_s = criterion(outputs_s,s_["label"])
            if not fn:
                meter_s.update_loss(loss_s.detach().cpu().numpy())
                meter_s.update_outputs(outputs_s.detach().cpu().numpy(), s_["label"].cpu().numpy())
            else:
                meter_s.add(s_["spectrum"],
                        s_["label"].detach().cpu().numpy(),
                        s_["channel_name"],
                        s_["start_end"],
                        s_["intensity"], 
                        s_["waveform"],
                        outputs_s.detach().cpu().numpy())
    acc_a = meter_a.accuracy()
    acc_s = meter_s.accuracy()            

    if fn is not None:
        loss_a = 0
        loss_s = 0
        meter_a.dump_prediction(os.path.join(fn, "artifacts.pkl"))
        meter_s.dump_prediction(os.path.join(fn, "spikes.pkl"))
    else:
        loss_s = meter_s.loss()
        loss_a = meter_a.loss()
    print('Inference: Time %.3f, loss_a: %.3f, accuracy_a: %.3f, loss_s: %.3f, accuracy_s-`: %.3f' %
        (time.time()-start,loss_a, acc_a, loss_s, acc_s))

    f1_s = meter_s.f1()
    f1_a = meter_a.f1()
    return loss_s, acc_s,f1_s ,loss_a, acc_a, f1_a


def create_sa_labels(spectrum, waveform, intensity, label, channel_name, start_end, computing_device):
    channel_name = np.array(channel_name)
    spectrum_norm = normalize_img(spectrum)
    inputs_a = torch.stack([spectrum_norm,spectrum_norm,spectrum_norm], dim=1, out=None).to(computing_device).float()  
    label = label.squeeze().float()
     
    select_index = torch.nonzero(label, as_tuple=False).squeeze(1)
    if label.dim() == 0:
        label = label.unsqueeze(0)
    train_spike_bool = len(select_index) > 1 
    if train_spike_bool:
        label_spike = label[select_index].clone() - 1
        label[select_index] = 1
    label = label.to(computing_device) 

    a_ = {
        "inputs": expand_dim(inputs_a, 4),
        "spectrum":spectrum,
        "label": expand_dim(label,2), 
        "intensity":intensity,
        "waveform":waveform,
        "channel_name":channel_name, 
        "start_end":start_end
    }

    s_ ={}    

    ## second
    if not train_spike_bool:
        return  a_, s_ ,False
   
    s_spectrum = spectrum[select_index]
    s_spectrum_norm = normalize_img(s_spectrum)
    intensity_norm = normalize_img(intensity[select_index])
    inputs_s = torch.stack([s_spectrum_norm,waveform[select_index],intensity_norm], dim=1, out=None).to(computing_device).float()
    label_s= label_spike.to(computing_device)
    s_ = {
        "inputs": expand_dim(inputs_s, 4),
        "spectrum":spectrum[select_index],
        "label": expand_dim(label_s, 1), 
        "intensity":intensity[select_index],
        "waveform":waveform[select_index],
        "channel_name":channel_name[select_index], 
        "start_end":start_end[select_index]
    }
    
    return a_, s_ , True
    
def train_model(model, train_loader,val_loader, test_loader ,criterion, optimizer, computing_device ,num_epochs_s=10,num_epochs_a=5,  stats_folder=None):
    since = time.time()
    best_loss_s = 100
    best_loss_a = 100
    checkpoint_folder = stats_folder
    if not os.path.exists(checkpoint_folder):
        os.mkdir(checkpoint_folder)

    optimizer_s = optimizer['spike']
    optimizer_a = optimizer['artifact']
    model_s = model["spike"]
    model_a = model["artifact"]

    best_model_s = None
    best_model_a = None
    
    best_loss_a_real = best_loss_a
    count = 0
    train_a = True
    
    train_meter_s = TrainingMeter("spike")
    train_meter_a = TrainingMeter("artifact")

    t_acc_s, t_f1_a, t_f1_s, t_acc_a = 0, 0, 0, 0
    for epoch in range(num_epochs_s):
        M = 1
        print('-' * 10)
        epoch_loss = 0
        
        model_a.train()
        model_s.train()

        meter_s = Meter("spike")
        meter_a = Meter("artifact")
        
        for _, (spectrum, waveform, intensity, label, info, start_end) in enumerate(train_loader, 0):
            a_, s_ , train_s = create_sa_labels(spectrum, waveform, intensity ,label, info, start_end, computing_device)
            if train_a:
                a_["label"] = a_["label"].squeeze()
                optimizer_a.zero_grad()

                outputs_a = model_a(a_["inputs"]).squeeze(1)
                loss_a = criterion(outputs_a, a_["label"])
                
                meter_a.update_loss(loss_a.detach().cpu().numpy())
                meter_a.update_outputs(outputs_a.detach().cpu().numpy(), a_["label"].cpu().numpy())

                loss_a = torch.sum(loss_a) *1.0 / len(outputs_a)
                loss_a.backward()
                optimizer_a.step()
            
            ## second
            if not train_s:
                continue
            optimizer_s.zero_grad()
            outputs_s = model_s(s_["inputs"])
            outputs_s = outputs_s.squeeze(1)
            s_["label"] = s_["label"]
            loss_s = criterion(outputs_s,s_["label"])
            meter_s.update_loss(loss_s.detach().cpu().numpy())
            meter_s.update_outputs(outputs_s.detach().cpu().numpy(), s_["label"].cpu().numpy())
            loss_s = torch.sum(loss_s) *1.0 / len(outputs_s)
            loss_s.backward()
            optimizer_s.step()
            
        # Print the loss averaged over the last N mini-batches
        if train_a:   
            loss_a = meter_a.loss()
            loss_s = meter_s.loss()
            acc_a = meter_a.accuracy()
            acc_s = meter_s.accuracy()
        else:
            loss_a = 0
            loss_s = meter_s.loss()
            acc_a = 0
            acc_s = meter_s.accuracy()
        print('Epoch %d, loss_a: %.3f, accuracy_a: %.3f , loss_s: %.3f, accuracy_s: %.3f' %
            (epoch + 1, loss_a, acc_a, loss_s, acc_s))

        #Validation
        if epoch % M == 0 and epoch!=0: 
            v_loss_s, v_acc_s, v_f1_s, v_loss_a, v_acc_a, v_f1_a  = validate(val_loader,{"artifact": model_a, "spike": model_s}, criterion, computing_device)  
           
            best_loss_s, best_model_s = pick_best_model(model_s,best_model_s ,epoch, v_loss_s, best_loss_s, checkpoint_folder, model_name="s")
            best_loss_a, best_model_a = pick_best_model(model_a,best_model_a ,epoch, v_loss_a, best_loss_a, checkpoint_folder, model_name="a")
            if best_loss_a == best_loss_a_real:
                count += 1
                if count >= 2:
                    train_a = False
            else:
                best_loss_a_real = best_loss_a
                count = 0
            #print("----test_-----")
            #t_loss_s, t_acc_s, t_f1_s , t_loss_a, t_acc_a, t_f1_a = validate(test_loader,{"artifact": model_a, "spike": model_s}, criterion, computing_device)  
            train_meter_s.add(acc_s,loss_s, v_loss_s,v_acc_s, 0, v_f1_s, t_acc_s, t_f1_s)   
            train_meter_a.add(acc_a,loss_a, v_loss_a,v_acc_a, 0, v_f1_a, t_acc_a, t_f1_a)           
    print("Training complete after", epoch + 1, "epochs")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s '.format(
        time_elapsed // 60, time_elapsed % 60))

    train_meter_s.dump_pickle(os.path.join(stats_folder, "training_curve_s.pkl"))
    train_meter_a.dump_pickle(os.path.join(stats_folder, "training_curve_a.pkl"))

    return {"artifact": best_model_a, "spike": best_model_s}

def pipeline(args, test_set_index=-1, k_fold= -1):
    if (test_set_index == -1 and k_fold == -1) or (test_set_index != -1 and k_fold != -1):
        raise NotImplemented

    model_artifact = NeuralCNN(num_classes=2)
    model_spike = NeuralCNN(num_classes=2)

    data_dir = args.data_dir
    res_dir = os.path.join(args.work_dir, args.res_dir)        
    os.makedirs(res_dir, exist_ok=True)
    num_epochs_s = args.num_epochs_s       # Number of full passes through the dataset
    num_epochs_a = args.num_epochs_a 
    batch_size = args.batch_size         # Number of samples in each minibatch
    learning_rate_a = args.learning_rate_a   ###0.001, 0.0005
    learning_rate_s = args.learning_rate_s
    seed = args.seed                # Seed the random number generator for reproducibility
    p_val = args.p_val              # Percent of the overall dataset to reserve for validation
    p_test = args.p_test             # Percent of the overall dataset to reserve for testing

    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device(args.device)
        extras = {"num_workers": 1, "pin_memory": True}
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")

    model_artifact = model_artifact.to(computing_device)
    model_spike = model_spike.to(computing_device)
    model = {"artifact": model_artifact, "spike": model_spike}

    criterion = nn.BCELoss(reduction="none").to(computing_device)
    optimizer_artifact = optim.Adam(filter(lambda p: p.requires_grad, model_artifact.parameters()), lr=learning_rate_a)
    optimizer_spike = optim.Adam(filter(lambda p: p.requires_grad, model_spike.parameters()), lr=learning_rate_s)
    optimizer = {"artifact": optimizer_artifact, "spike":optimizer_spike}

    print("Model on CUDA?", next(model_spike.parameters()).is_cuda)    

    start_time = time.time()
    if k_fold != -1:
        train_loader, val_loader, test_loader = create_kfold_loader(data_dir,k_fold,batch_size, seed, 
                                                                    p_val=p_val, p_test=p_test,shuffle=True, extras=extras)
        stats_folder = os.path.join(res_dir, "overall")
        if not os.path.exists(stats_folder):
            os.mkdir(stats_folder)
        stats_folder = os.path.join(stats_folder, str(k_fold))   
        print("Kfold", k_fold)
    elif test_set_index !=-1:
        train_loader, val_loader, test_loader = create_patient_eliminate_loader(data_dir, test_set_index,batch_size, seed=seed,
                                                                        p_val=p_val, p_test=p_test, shuffle=True, extras=extras)   
        patient_names = sorted(os.listdir(data_dir))[test_set_index]
        stats_folder = os.path.join(res_dir,patient_names)          
    else:
        raise NotImplemented                                     
    clean_folder(stats_folder)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Prepare dataset | Time: {epoch_mins}m {epoch_secs}s')    

    print("----------------Training----------------")
    model_trained = train_model(model, train_loader, val_loader, test_loader ,criterion, optimizer, computing_device,
    num_epochs_s=num_epochs_s, num_epochs_a=num_epochs_a, stats_folder=stats_folder)
    
    print("-----------------testing ----------------")
    #loss_s, acc_s, _ ,loss_a, acc_a, _ = validate(test_loader, model_trained,criterion, computing_device, fn=stats_folder)
    loss_s, acc_s, _ ,loss_a, acc_a, _ = validate(test_loader, model_trained,criterion, computing_device, fn=stats_folder) 
    return loss_s, acc_s, loss_a, acc_a

def patient_fold(args):
    patient_names = sorted(os.listdir(args.data_dir))
    print([f"{i}, {p_name}" for i, p_name in enumerate(patient_names)])

    pid_list = range(len(patient_names))
    for p_idx in pid_list:
        loss_s, acc_s, loss_a, acc_a = pipeline(args, test_set_index = p_idx)

def all_patient(args):
    for k in range(args.num_k):
        k = k + 1 
        print("--------------------- Fold: " + str(k) + "-----------------------")
        loss_s, acc_s, loss_a, acc_a = pipeline(args, k_fold = k)
    
if __name__ == "__main__":
    args = arg_parse(sys.argv[1:])
    print(args)

    if args.all_patient:
        all_patient(args)
    else:
        patient_fold(args)
