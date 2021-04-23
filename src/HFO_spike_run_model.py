#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch
import os
from PIL import Image
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F 
from utilities import *
import quickdraw_dataloader as qd
from quickdraw_dataloader import create_split_loaders, create_split_loaders_kfold
import torchvision.models as models


def accuracy(out, labels):
    out = np.squeeze(out) > 0.5
    labels = np.squeeze(labels)
    correct = out==labels
    return 1.0*np.sum(out==labels)/float(labels.size)


def validate(val_loader,model,criterion, computing_device):
    start = time.time()
    sum_loss = 0.0
    list_sum_loss = []
    outputs_all = []
    labels_all = []
    num = 0
    for mb_count, (val_images, val_labels) in enumerate(val_loader, 0):
        model.eval()
        with torch.no_grad():       
            val_images = torch.squeeze(torch.stack([val_images,val_images,val_images], dim=1, out=None)).float()
            val_images, val_labels = val_images.to(computing_device), val_labels.to(computing_device).squeeze()
            val_labels = val_labels.type(torch.cuda.FloatTensor) -1
            outputs = model(val_images).squeeze()
            loss = criterion(outputs,val_labels)
            sum_loss += 1.0*loss.item()
            outputs_all.append(outputs.cpu().detach().numpy())
            labels_all.append(val_labels.cpu().detach().numpy())
    #print("after validation: ", outputs_all)
    outputs_all = np.concatenate(outputs_all)
    labels_all = np.concatenate(labels_all)
    accuracy_val = accuracy(outputs_all, labels_all)
    val_loss = 1.0*sum_loss/len(val_loader)
    print('Validation Time %.3f, loss: %.3f, accuracy_val: %.3f' %
        (time.time()-start, val_loss, accuracy_val))    
    return val_loss , accuracy_val




def train_model(model, train_loader,val_loader ,criterion, optimizer, computing_device, p_index ,num_epochs=10):
    since = time.time()
    total_loss = []
    avg_minibatch_loss = []
    total_vali_loss = []
    tolerence = 3
    i = 0 
    best_loss = 100
    for epoch in range(num_epochs):
        model.train()
        M = 1
        N_minibatch_loss = 0.0    
        early_stop = 0
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_loss = 0
        # Each epoch has a training and validation phase
        for phase in ['train']:
            # Iterate over data.
            outputs_all = []
            labels_all = []
            model.train()
            for minibatch_count, (inputs, labels) in enumerate(train_loader, 0):
                inputs = torch.squeeze(torch.stack([inputs,inputs,inputs], dim=1, out=None)).to(computing_device).float()
                labels = labels.to(computing_device).float().squeeze() -1
                optimizer.zero_grad()

                outputs = model(inputs).squeeze()
                loss = criterion(outputs,labels)
                
                loss.backward()
                optimizer.step()

                epoch_loss = epoch_loss + loss.item()
                output_np = outputs.cpu().detach().numpy()
                label_np = labels.cpu().detach().numpy()
                outputs_all.append(output_np)
                labels_all.append(label_np)

        # Print the loss averaged over the last N mini-batches    
        epoch_loss = 1.0*epoch_loss/len(train_loader)

        accuracy_train = accuracy(np.concatenate(outputs_all), np.concatenate(labels_all))
        print('Epoch %d, loss: %.3f, accuracy_train: %.3f' %
            (epoch + 1, epoch_loss, accuracy_train))

        #Validation
        if epoch % M == 0 and epoch!=0: 
            #model = torch.load('./checkpoint')
            patient_names = sorted(os.listdir(root_dir))[p_index]
            v_loss, v_acc = validate(val_loader,model.eval(),criterion, computing_device)                  

            if v_loss < best_loss:
                best_loss = v_loss 
                best_model = model
                print("best_model in epoch ", epoch +1)
                save_checkpoint({'epoch': epoch + 1,
                        'state_dict': best_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        },
                        filename='./checkpoint_crop/spike_hfo/'+patient_names+'_model.pth')
                        #filename='./checkpoint/'+patient_names+'_model.pth')
    print("Training complete after", epoch, "epochs")
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s '.format(
        time_elapsed // 60, time_elapsed % 60))
    return best_model


def save_checkpoint(state, is_best=0, filename='models/checkpoint.pth.tar'):
    torch.save(state, filename)


def pipeline(test_set_index, root_dir):
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(512, 1),nn.Sigmoid() )


    num_epochs = 15          # Number of full passes through the dataset
    batch_size = 128         # Number of samples in each minibatch
    learning_rate = 0.0008   ###0.001
    seed = np.random.seed(0) # Seed the random number generator for reproducibility
    p_val = 0.1              # Percent of the overall dataset to reserve for validation
    p_test = 0.2             # Percent of the overall dataset to reserve for testing

    transform = transforms.Compose([
            #transforms.RandomResizedCrop(224),
            transforms.ToPILImage(mode=None),
            transforms.Resize([224,224],interpolation=2),
            transforms.ToTensor(),
            #transforms.Normalize([0.5], [0.229, 0.224, 0.225])
        ])

    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda:0")
        extras = {"num_workers": 16, "pin_memory": True}
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")

    model = model
    model = model.to(computing_device)
    print("Model on CUDA?", next(model.parameters()).is_cuda)    

    start_time = time.time()
    train_loader, val_loader, test_loader = create_split_loaders_kfold(root_dir, test_set_index,batch_size, seed, transform=transform, 
                                                                p_val=p_val, p_test=p_test,
                                                                shuffle=True, show_sample=False, 
                                                                extras=extras)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Prepare dataset | Time: {epoch_mins}m {epoch_secs}s')    

    criterion = nn.BCELoss().to(computing_device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    model_trained = train_model(model, train_loader, val_loader ,criterion, optimizer, computing_device, test_set_index ,num_epochs=num_epochs)
    print("-----------------testing ----------------")
    patient_names = sorted(os.listdir(root_dir))[test_set_index]
    print("patient_names is", patient_names)
    loss, acc = validate(test_loader, model_trained,criterion, computing_device)
    return loss, acc


# if __name__ == "__main__":
#     root_dir = "/mnt/SSD2/qiujing_data/HFO_classification_training_data_crop"
#     p_name = []
#     p_acc = []
#     patient_names = sorted(os.listdir(root_dir))
#     print(patient_names)
#     for p_idx, pn in enumerate(patient_names):
#         loss, acc = pipeline(p_idx, root_dir)
#         p_name.append(pn)
#         p_acc.append(acc)
#         res = {"name":p_name, "acc":p_acc}
#         dump_pickle("stats.pkl", res)

if __name__ == "__main__":
    root_dir = "/mnt/SSD2/qiujing_data/HFO_classification_training_data_spike"
    p_name = []
    p_acc = []
    patient_names = sorted(os.listdir(root_dir))
    print([f"{i}, {p_name}" for i, p_name in enumerate(patient_names)])
    pid_list = range(len(patient_names))
    for p_idx in pid_list:
        loss, acc = pipeline(p_idx, root_dir)
        pn = patient_names[p_idx]
        p_name.append(pn)
        p_acc.append(acc)
        #res = {"name":p_name, "acc":p_acc}
        #dump_pickle("stats.pkl", res)

