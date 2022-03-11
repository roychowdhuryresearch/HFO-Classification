# PyTorch imports
from cmath import inf
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

# Other libraries for data manipulation and visualization
import os
from PIL import Image
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as data
import os 

class HFODataset(Dataset):

    def __init__(self, data_dir, patient_name ,transform=transforms.ToTensor(), remove_artifacts=False):

        spectrum_folder = os.path.join(data_dir, patient_name)
        
        loaded = np.load(os.path.join(spectrum_folder,"data.npz"), allow_pickle=True)
        self.spectrum = np.squeeze(loaded["spectrum"])
        label_fn = os.path.join(spectrum_folder,"label.npz")
        try:
            if os.path.exists(label_fn):
                self.label = np.load(label_fn)["labels"].astype(int)
            else:
                self.label = np.zeros(len(self.spectrum)).astype(int)
                print("here",label_fn)
        except:
            self.label = np.zeros(len(self.spectrum)).astype(int)
            print("here1",label_fn)
        self.label = self.label.reshape(-1, 1)
        self.info = np.squeeze(loaded["info"])
        self.start_end = np.squeeze(loaded["start_end"]).reshape(-1, 2).astype(int)
        self.intensity = loaded["intensity"]
        self.waveform = loaded["waveform"]
        print(self.spectrum.shape, self.label.shape, self.start_end.shape, self.intensity.shape, self.waveform.shape, self.info.shape)
        if remove_artifacts:
            self.__remove_artifacts()

        self.length = len(self.label)        
        self.transform = transform
       
    def __remove_artifacts(self):
        non_artifacts_index = np.where(self.label > 0)[0]  
        self.spectrum = self.spectrum[non_artifacts_index]
        self.label = self.label[non_artifacts_index]-1
        self.info = self.info[non_artifacts_index]
        self.start_end =  self.start_end[non_artifacts_index]
        self.intensity =  self.intensity[non_artifacts_index]
        self.waveform =  self.waveform[non_artifacts_index]

    def __len__(self):
        
        # Return the total number of data samples
        return self.length


    def __getitem__(self, ind):
        """Returns the image and its label at the index 'ind' 
        (after applying transformations to the image, if specified).
        
        Params:
        -------
        - ind: (int) The index of the image to get

        Returns:
        --------
        - A tuple (image, label)
        """
        spectrum = torch.from_numpy(self.spectrum[ind])
        label = torch.from_numpy(np.array(self.label[ind]))
        info = self.info[ind]
        start_end = torch.from_numpy(self.start_end[ind])
        waveform = torch.from_numpy(np.rot90(self.waveform[ind]).copy())
        intensity = torch.from_numpy(self.intensity[ind])
        return spectrum, waveform, intensity, label, info, start_end


def create_patient_eliminate_loader(data_dir, test_set_index,batch_size, seed=0, transform=transforms.ToTensor(),
                         p_val=0.2, p_test=0.2, shuffle=True, extras={}):
     
    list_of_datasets = []
    for j in sorted(os.listdir(data_dir)):
        list_of_datasets.append(HFODataset(data_dir=data_dir, patient_name=j, transform=transform))
    # once all single json datasets are created you can concat them into a single one:
    hfo_dataset = data.ConcatDataset(list_of_datasets)
    
    testing_set = list_of_datasets[test_set_index]
    list_of_datasets.pop(test_set_index)
    
    training_set = data.ConcatDataset(list_of_datasets)
    dataset_size = len(training_set)
    all_indices = list(range(dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)
    
    # Create the validation split from the full dataset
    val_split = int(np.floor(p_val * dataset_size))
    train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
    
    val_split = int(np.floor(p_val * dataset_size))
    train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]
    
    sample_train = SubsetRandomSampler(train_ind)
    sample_val = SubsetRandomSampler(val_ind)
    
    train_loader = DataLoader(training_set, batch_size=batch_size, 
                            sampler=sample_train, num_workers=num_workers, 
                            pin_memory=pin_memory)
    val_loader = DataLoader(training_set, batch_size=batch_size,
                        sampler=sample_val, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    test_loader = DataLoader(testing_set, batch_size=batch_size, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader)


def create_patient_loader(data_dir, patient_name, batch_size, seed=0, transform=transforms.ToTensor(),
                         p_val=0.2, p_test=0.2, shuffle=True, extras={}):
     
    testing_set = HFODataset(data_dir=data_dir, patient_name=patient_name, transform=transform, remove_artifacts= True) 

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
     
    test_loader = DataLoader(testing_set, batch_size=batch_size, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    # Return the training, validation, test DataLoader objects
    return (test_loader)


def create_kfold_loader(data_dir, folder_num, batch_size, seed=0, transform=transforms.ToTensor(),
                         p_val=0.1, p_test=0.2, shuffle=True, extras={}):
    list_of_datasets = []
    for j in sorted(os.listdir(data_dir)):
        list_of_datasets.append(HFODataset(data_dir=data_dir, patient_name=j, transform=transform))
    # once all single json datasets are created you can concat them into a single one:
    hfo_dataset = data.ConcatDataset(list_of_datasets)
    
    dataset_size = len(hfo_dataset)
    
    all_indices = list(range(dataset_size))
    # Create the validation split from the full dataset
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)
    test_split = int(np.floor(p_test * dataset_size))
    test_start = (folder_num - 1)*test_split
    test_end = test_start + test_split
    test_ind = all_indices[test_start: test_end]
    remain_ind = list(set(all_indices) - set(test_ind))
    fixed_val_percent = int(np.floor(p_val * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(remain_ind)
    val_ind, train_ind = remain_ind[:fixed_val_percent] , remain_ind[fixed_val_percent:]

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    sample_train = SubsetRandomSampler(train_ind)
    sample_val = SubsetRandomSampler(val_ind)
    sample_test = SubsetRandomSampler(test_ind)
    
    train_loader = DataLoader(hfo_dataset, batch_size=batch_size, 
                            sampler=sample_train, num_workers=num_workers, 
                            pin_memory=pin_memory)
    val_loader = DataLoader(hfo_dataset, batch_size=batch_size,
                        sampler=sample_val, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    test_loader = DataLoader(hfo_dataset,sampler=sample_test,batch_size=batch_size, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader)

