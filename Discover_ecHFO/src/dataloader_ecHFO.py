# PyTorch imports
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

# Other libraries for data manipulation and visualization
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as data
import os 
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
from patient_info import seizure_free_patient_names, patient_resected, patient_90

# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(0)
# random.seed(0)

def min_max(x):
    xx = x.reshape(len(x), -1)
    x = (x - xx.min(axis=-1)[:, None, None]) / (xx.max(axis=-1)[:, None, None] - xx.min(axis=-1)[:, None, None])
    return x

class HFODataset(Dataset):
    """Custom Dataset class for the Chest X-Ray Dataset.

    The expected dataset is stored in the "/datasets/ChestXray-NIHCC/" on ieng6
    """
    
    def __init__(self, data_dir, patient_name ,transform=transforms.ToTensor(), mode= "Model1", filter_both= False, keep= False):
        #print(patient_name)
        self.patient_name = patient_name
        spectrum_folder = os.path.join(data_dir, patient_name)
        self.mode = mode

        loaded = np.load(os.path.join(spectrum_folder,"data.npz"), allow_pickle=True)    
        label_load = np.load(os.path.join(spectrum_folder,"label_"+ mode +".npz"), allow_pickle=True)            
        
        self.spectrum = min_max(np.squeeze(loaded["spectrum"]))*255
        self.intensity = loaded["intensity"]
        self.intensity = min_max(self.intensity)*255
        self.waveform = loaded["waveform"]

        self.labels = np.squeeze(label_load["labels"].astype(int))
        self.artifacts_label = label_load["artifacts"]
        self.info = np.squeeze(loaded["info"])
        self.start_end = np.squeeze(loaded["start_end"]).reshape(-1, 2).astype(int)
       
        self._remove_artifacts()
        if filter_both:
            self._remove()
        if keep:
            self._keep()
        self.length = len(self.start_end)        
        self.transform = transform
        self.print_stats()
    
    def print_stats(self):
        print(self.patient_name, "  label1 : ", sum(self.labels), "  label0 : ",self.length - sum(self.labels))
    
    def _remove_artifacts(self):
        non_artifacts_index = np.where(self.artifacts_label > 0)[0]  
        self.spectrum = self.spectrum[non_artifacts_index]
        self.labels  =  self.labels[non_artifacts_index]
        self.info = self.info[non_artifacts_index]
        self.start_end =  self.start_end[non_artifacts_index]
        self.intensity =  self.intensity[non_artifacts_index]
        self.waveform =  self.waveform[non_artifacts_index]
    
    def _remove(self):
        index = np.where(self.labels != -1)[0]  
        self.spectrum = self.spectrum[index]
        self.info = self.info[index]
        self.start_end =  self.start_end[index]
        self.intensity =  self.intensity[index]
        self.waveform =  self.waveform[index]
        self.labels = self.labels[index]
    
    def _keep(self):
        index = np.where(self.labels == -1)[0]  
        self.spectrum = self.spectrum[index]
        self.info = self.info[index]
        self.start_end =  self.start_end[index]
        self.intensity =  self.intensity[index]
        self.waveform =  self.waveform[index]
        self.labels = self.labels[index]

    def __len__(self):
        # Return the total number of data samples
        return self.length

    def __getitem__(self, ind):
        spectrum = torch.from_numpy(self.spectrum[ind])
        label = torch.from_numpy(np.array([self.labels[ind], self.labels[ind]]))
        info = self.info[ind]
        start_end = np.array(self.start_end[ind])
        waveform =  torch.from_numpy(np.rot90(self.waveform[ind]).copy())
        intensity =  torch.from_numpy(self.intensity[ind])
        return self.patient_name, spectrum, waveform, intensity, label, info, start_end

def create_split_loaders_overall(data_dir, test_set_index,batch_size, seed=0, transform=transforms.ToTensor(),
                         p_val=0.2, p_test=0.2, shuffle=True, 
                         show_sample=False, extras={}):
    list_of_datasets = []
    print(os.listdir(data_dir))
    for j_inx, j in enumerate(os.listdir(data_dir)):
        if j not in patient_90:
            continue
        list_of_datasets.append(HFODataset(data_dir=data_dir, patient_name=j, transform=transform))
    # once all single datasets are created you can concat them into a single one:
    hfo_dataset = data.ConcatDataset(list_of_datasets)
    
    dataset_size = len(hfo_dataset)
    all_indices = list(range(dataset_size))
    if shuffle:
        #torch.manual_seed(0)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        random.seed(0)
        np.random.shuffle(all_indices)
    
    # Create the validation split from the full dataset
    val_split = int(np.floor(p_val * dataset_size))
    test_split = int(np.floor(p_test * dataset_size))

    num_workers = 1
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
    
    val_split = int(np.floor(p_val * dataset_size))
    train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]
    train_ind, test_ind = train_ind[test_split :], train_ind[: test_split]
    
    sample_train = SubsetRandomSampler(train_ind)
    sample_val = SubsetRandomSampler(val_ind)
    sample_test = SubsetRandomSampler(test_ind)
    
    train_loader = DataLoader(hfo_dataset, batch_size = batch_size, 
                            sampler=sample_train, num_workers=num_workers, 
                            pin_memory=pin_memory)
    val_loader = DataLoader(hfo_dataset, batch_size=batch_size,
                        sampler=sample_val, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    test_loader = DataLoader(hfo_dataset,sampler=sample_test,batch_size=batch_size, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    # Return the training, validation, test  objects
    return (train_loader, val_loader, test_loader)


def create_split_loaders_overall(data_dir, test_set_index,batch_size, seed=0, transform=transforms.ToTensor(),
                         p_val=0.2, p_test=0.2, shuffle=True, 
                         show_sample=False, extras={}):
    """ Creates the  objects for the training, validation, and test sets. 

    Params:
    -------
    - batch_size: (int) mini-batch size to load at a time
    - seed: (int) Seed for random generator (use for testing/reproducibility)
    - transform: A torchvision.transforms object - transformations to apply to each spectrum
                 (Can be "transforms.Compose([transforms])")
    - p_val: (float) Percent (as decimal) of dataset to use for validation
    - p_test: (float) Percent (as decimal) of the dataset to split for testing
    - shuffle: (bool) Indicate whether to shuffle the dataset before splitting
    - show_sample: (bool) Plot a mini-example as a grid of the dataset
    - extras: (dict) 
        If CUDA/GPU computing is supported, contains:
        - num_workers: (int) Number of subprocesses to use while loading the dataset
        - pin_memory: (bool) For use with CUDA - copy tensors into pinned memory 
                  (set to True if using a GPU)
        Otherwise, extras is an empty dict.

    Returns:
    --------
    - train_loader: () The iterator for the training set
    - val_loader: () The iterator for the validation set
    - test_loader: () The iterator for the test set
    """
    list_of_datasets = []
    for j_inx, j in enumerate(os.listdir(data_dir)):
        if j not in patient_90:
        #if j not in patient_resected: ##just for sanity check 
            continue
        list_of_datasets.append(HFODataset(data_dir=data_dir, patient_name=j, transform=transform))
    # once all single datasets are created you can concat them into a single one:
    hfo_dataset = data.ConcatDataset(list_of_datasets)
    
    dataset_size = len(hfo_dataset)
    all_indices = list(range(dataset_size))
    if shuffle:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        random.seed(0)
        np.random.shuffle(all_indices)
    
    # Create the validation split from the full dataset
    val_split = int(np.floor(p_val * dataset_size))
    test_split = int(np.floor(p_test * dataset_size))

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
    
    val_split = int(np.floor(p_val * dataset_size))
    train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]
    train_ind, test_ind = train_ind[test_split :], train_ind[: test_split]
    
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
    
    # Return the training, validation, test  objects
    return (train_loader, val_loader, test_loader)


def create_patient_fold(data_dir, test_patient_name,batch_size, seed=0, transform=transforms.ToTensor(),
                         p_val=0.2, p_test=0.2, shuffle=True, 
                         show_sample=False, extras={}):
    """ Creates the  objects for the training, validation, and test sets. 

    Params:
    -------
    - batch_size: (int) mini-batch size to load at a time
    - seed: (int) Seed for random generator (use for testing/reproducibility)
    - transform: A torchvision.transforms object - transformations to apply to each spectrum
                 (Can be "transforms.Compose([transforms])")
    - p_val: (float) Percent (as decimal) of dataset to use for validation
    - p_test: (float) Percent (as decimal) of the dataset to split for testing
    - shuffle: (bool) Indicate whether to shuffle the dataset before splitting
    - show_sample: (bool) Plot a mini-example as a grid of the dataset
    - extras: (dict) 
        If CUDA/GPU computing is supported, contains:
        - num_workers: (int) Number of subprocesses to use while loading the dataset
        - pin_memory: (bool) For use with CUDA - copy tensors into pinned memory 
                  (set to True if using a GPU)
        Otherwise, extras is an empty dict.

    Returns:
    --------
    - train_loader: () The iterator for the training set
    - val_loader: () The iterator for the validation set
    - test_loader: () The iterator for the test set
    """
    list_of_datasets = []
    print(os.listdir(data_dir))
    for j_inx, j in enumerate(os.listdir(data_dir)):
        if j == test_patient_name or j not in patient_90:
            continue
        list_of_datasets.append(HFODataset(data_dir=data_dir, patient_name=j, transform=transform))
    # once all single datasets are created you can concat them into a single one:
    hfo_dataset = data.ConcatDataset(list_of_datasets)
    
    dataset_size = len(hfo_dataset)
    all_indices = list(range(dataset_size))
    if shuffle:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        random.seed(0)
        np.random.shuffle(all_indices)
    
    # Create the validation split from the full dataset
    val_split = int(np.floor(p_val * dataset_size))

    num_workers = 1
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
    
    val_split = int(np.floor(p_val * dataset_size))
    train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]
    
    sample_train = SubsetRandomSampler(train_ind)
    sample_val = SubsetRandomSampler(val_ind)

    test_set = HFODataset(data_dir=data_dir, patient_name=test_patient_name, transform=transform, filter_data= False)
    
    train_loader = DataLoader(hfo_dataset, batch_size=batch_size, 
                            sampler=sample_train, num_workers=num_workers, 
                            pin_memory=pin_memory)
    val_loader = DataLoader(hfo_dataset, batch_size=batch_size,
                        sampler=sample_val, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    test_loader = DataLoader(test_set,batch_size=batch_size, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    # Return the training, validation, test  objects
    return (train_loader, val_loader, test_loader)

def create_patient_loader_90(data_dir, patient_name, batch_size, seed=0, transform=transforms.ToTensor(),
                         p_val=0.2, p_test=0.2, shuffle=True, extras={}):
     
    testing_set = HFODataset(data_dir=data_dir, patient_name=patient_name, transform=transform, filter_data=False) 

    num_workers = 1
    pin_memory = True
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
     
    test_loader = DataLoader(testing_set, batch_size=batch_size, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    # Return the training, validation, test  objects
    return (test_loader)
