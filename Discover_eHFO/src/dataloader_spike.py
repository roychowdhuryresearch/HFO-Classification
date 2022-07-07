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
from patient_info import seizure_free_patient_names, patient_resected


class HFODataset(Dataset):
    """Custom Dataset class for the Chest X-Ray Dataset.

    The expected dataset is stored in the "/datasets/ChestXray-NIHCC/" on ieng6
    """
    
    def __init__(self, data_dir, patient_name ,transform=transforms.ToTensor(), filter_data = True):
        #print(patient_name)
        spectrum_folder = os.path.join(data_dir, patient_name)
      
        loaded = np.load(os.path.join(spectrum_folder,"data.npz"))
        self.spectrum = np.squeeze(loaded["spectrum"])
        #self.label = np.squeeze(load_pickle(os.path.join(spectrum_folder, "artifacts_pred.pkl")))
        label_load = np.load(os.path.join(spectrum_folder,"label.npz"))
        self.label_removed = label_load["remove_labels"]
        self.label_soz = label_load["soz_labels"]
        self.artifacts_label = label_load["artifacts"] 
        self.info = np.squeeze(loaded["info"])
        self.start_end = np.squeeze(loaded["start_end"])
        
        self.intensity = loaded["intensity"]
        self.waveform = loaded["waveform"]

        self._remove_artifacts()
        if filter_data:
            self._filter_data()
        self.length = len(self.label_removed)        
        self.transform = transform

    def _remove_artifacts(self):
        non_artifacts_index = np.where(self.artifacts_label > 0)[0]  
        self.spectrum = self.spectrum[non_artifacts_index]
        self.label_removed = self.label_removed[non_artifacts_index]
        self.label_soz =  self.label_soz[non_artifacts_index]
        self.info = self.info[non_artifacts_index]
        self.start_end =  self.start_end[non_artifacts_index]
        self.intensity =  self.intensity[non_artifacts_index]
        self.waveform =  self.waveform[non_artifacts_index]

    def _filter_data(self, keep=2500):
        if len(self.label_removed) <= 2500:
            return
        total_index = np.arange(len(self.label_removed))
        np.random.shuffle(total_index)
        remain_index = total_index[:keep]
        #print(np.mean(remain_index), np.median(remain_index), np.var(remain_index))
        self.spectrum = self.spectrum[remain_index]
        self.label_removed = self.label_removed[remain_index]
        self.label_soz =  self.label_soz[remain_index]
        self.info = self.info[remain_index]
        self.start_end =  self.start_end[remain_index]
        self.intensity =  self.intensity[remain_index]
        self.waveform =  self.waveform[remain_index]

    def __len__(self):
        
        # Return the total number of data samples
        return self.length


    def __getitem__(self, ind):
        """Returns the spectrum and its label at the index 'ind' 
        (after applying transformations to the spectrum, if specified).
        
        Params:
        -------
        - ind: (int) The index of the spectrum to get

        Returns:
        --------
        - A tuple (spectrum, label)
        """
        spectrum = torch.from_numpy(self.spectrum[ind])
        #spectrum[215:,:] = 0
        label = torch.from_numpy(np.array([self.label_removed[ind], self.label_soz[ind]]))
        info = self.info[ind]
        start_end = np.array(self.start_end[ind])
        waveform =  torch.from_numpy(np.rot90(self.waveform[ind]).copy())
        intensity =  torch.from_numpy(self.intensity[ind].copy())
        return spectrum, waveform, intensity, label, info, start_end

def create_split_loaders_overall(data_dir, test_set_index,batch_size, seed=0, transform=transforms.ToTensor(),
                         p_val=0.2, p_test=0.2, shuffle=True, 
                         show_sample=False, extras={}):
    """ Creates the DataLoader objects for the training, validation, and test sets. 

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
    - train_loader: (DataLoader) The iterator for the training set
    - val_loader: (DataLoader) The iterator for the validation set
    - test_loader: (DataLoader) The iterator for the test set
    """
    list_of_datasets = []
    for j_inx, j in enumerate(os.listdir(data_dir)):
        if j not in seizure_free_patient_names:
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
    
    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader)


def create_split_loaders_overall(data_dir, test_set_index,batch_size, seed=0, transform=transforms.ToTensor(),
                         p_val=0.2, p_test=0.2, shuffle=True, 
                         show_sample=False, extras={}):
    """ Creates the DataLoader objects for the training, validation, and test sets. 

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
    - train_loader: (DataLoader) The iterator for the training set
    - val_loader: (DataLoader) The iterator for the validation set
    - test_loader: (DataLoader) The iterator for the test set
    """
    list_of_datasets = []
    for j_inx, j in enumerate(os.listdir(data_dir)):
        if j not in seizure_free_patient_names:
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
    
    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader)


def create_patient_fold(data_dir, test_patient_name,batch_size, seed=0, transform=transforms.ToTensor(),
                         p_val=0.2, p_test=0.2, shuffle=True, 
                         show_sample=False, extras={}):
    """ Creates the DataLoader objects for the training, validation, and test sets. 

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
    - train_loader: (DataLoader) The iterator for the training set
    - val_loader: (DataLoader) The iterator for the validation set
    - test_loader: (DataLoader) The iterator for the test set
    """
    list_of_datasets = []
    for j_inx, j in enumerate(os.listdir(data_dir)):
        if j not in seizure_free_patient_names or j == test_patient_name:
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

    test_set = HFODataset(data_dir=data_dir, patient_name=test_patient_name, transform=transform, filter_data= False)
    
    train_loader = DataLoader(hfo_dataset, batch_size=batch_size, 
                            sampler=sample_train, num_workers=num_workers, 
                            pin_memory=pin_memory)
    val_loader = DataLoader(hfo_dataset, batch_size=batch_size,
                        sampler=sample_val, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    test_loader = DataLoader(test_set,batch_size=batch_size, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    # Return the training, validation, test DataLoader objects
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
    
    # Return the training, validation, test DataLoader objects
    return (test_loader)
