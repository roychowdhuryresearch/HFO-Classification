
import os, time, copy, sys

import torch
import torch.nn as nn
import torch.optim as optim
from random import random, sample
import numpy as np
from torch.utils.data import dataset, Subset, WeightedRandomSampler

from src.training_utils import *
from src.dataloader import HFODataset, read_data, parallel_process, SubsetRandomSampler
from src.model import NeuralCNN, PreProcessing
from src.meter import Meter
from torch.utils.data import  DataLoader, ConcatDataset
from src.args import args
import copy
import torch_pruning as tp
import pandas as pd
from tqdm import tqdm
class Trainer():
    def __init__(self, args, verbose=True):
        self.verbose = verbose
        self.data_dir = args["data_dir"]
        self.work_dir = args["work_dir"]
        self.res_dir = os.path.join(args["work_dir"], args["res_dir"], "ckpt")  #
        self.num_epochs = args["num_epochs"]  # Number of full passes through the dataset
        self.batch_size = args["batch_size"]  # Number of samples in each minibatch
        self.learning_rate = args["learning_rate"]
        self.seed = args["seed"]  # Seed the random number generator for reproducibility
        self.p_val = args["p_val"]  # Percent of the overall dataset to reserve for validation
        self.device = args["device"]

        os.makedirs(self.res_dir, exist_ok=True)
        self.criterion = nn.BCELoss(reduction="none").to(self.device)
        ## process 
        self.list_of_datasets = self.__construct_dataset()
        self.n_input = self.list_of_datasets[0].feature.shape[1]
        self.list_of_datasets_eval = [copy.deepcopy(d) for d in self.list_of_datasets]
        for i in range(len(self.list_of_datasets_eval)):
            self.list_of_datasets_eval[i].flip = False
        self.all_ds = ConcatDataset(self.list_of_datasets)
        self.all_ds_eval = ConcatDataset(self.list_of_datasets_eval)
        del self.list_of_datasets_eval
        del self.list_of_datasets
        self.augmentation_arg = args["augmentation_arg"]
        self.data_meta = pd.read_csv(os.path.join(self.data_dir, "data_meta.csv"))
        self.pre_processing = PreProcessing.from_df_args(self.data_meta,self.augmentation_arg)
        self.save_checkpoint = args["save_checkpoint"]

    def kfold_crossvalidation(self, K):
        for i in range(K):
            print('------------', i, 'th fold ------------')
            train_loader, val_loader, test_loader = self.__construct_training_valid_set_kfold(K, i)
            res_dir = os.path.join(self.res_dir, "fold_%d"%i)
            clean_folder(res_dir)
            model = self.train(train_loader, val_loader, res_dir)
            self.validate(test_loader, model, res_dir)
    
    def onefold_crossvalidation(self, K, k, model):
        train_loader, val_loader, test_loader = self.__construct_training_valid_set_kfold(K, k)
        res_dir = os.path.join(self.res_dir, "fold_%d"%k)
        os.makedirs(res_dir, exist_ok=True)
        model = self.train(train_loader, val_loader, res_dir, model)
        loss, acc = self.validate(test_loader, model, res_dir)
        return model, acc
    
    def test_onefold_crossvalidation(self, K, k, model):
        _ , _ , test_loader = self.__construct_training_valid_set_kfold(K, k)
        res_dir = os.path.join(self.res_dir, "fold_%d"%k)
        os.makedirs(res_dir, exist_ok=True)
        loss, acc  = self.validate(test_loader, model, res_dir)
        return model, acc

    def __initialize_model(self):
        model = NeuralCNN(in_channels=self.n_input, outputs=1).to(self.device).float()
        return model
    
    def __initialize_optimizer(self, model):
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate)

    def __construct_dataset(self):
        params = [{"data_dir":self.data_dir, "patient_name":patient_name} for patient_name in sorted(os.listdir(self.data_dir)) if os.path.isdir(os.path.join(self.data_dir, patient_name))]
        ret = parallel_process(params, read_data, 5, front_num=1, use_kwargs=True)
        list_dataset = []
        for r in ret:
            if isinstance(r, Exception):
                print(r)
                continue
            list_dataset.append(HFODataset(r))
        return list_dataset

    def __construct_training_valid_set_kfold(self, K, k):
        p_test = 1.0 / K
        all_ds = self.all_ds
        dataset_size = len(all_ds)
        all_indices = list(range(dataset_size))
        # Create the validation split from the full dataset
        np.random.seed(self.seed)
        np.random.shuffle(all_indices)
        test_split = int(np.floor( p_test* dataset_size))
        test_start = (k)*test_split
        test_end = test_start + test_split
        test_ind = all_indices[test_start: test_end]
        remain_ind = list(set(all_indices) - set(test_ind))
        fixed_val_percent = int(np.floor(self.p_val * dataset_size))
        np.random.seed(self.seed)
        np.random.shuffle(remain_ind)
        val_ind, train_ind = remain_ind[:fixed_val_percent] , remain_ind[fixed_val_percent:]
        train_set = Subset(all_ds, train_ind)
        val_set = Subset(self.all_ds_eval, val_ind)
        test_set = Subset(self.all_ds_eval, test_ind)
        num_workers = 1
        train_loader = DataLoader(train_set, batch_size=self.batch_size,  num_workers=num_workers, 
                            pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size,num_workers=num_workers, 
                                pin_memory=True)
    
        test_loader = DataLoader(test_set,batch_size=self.batch_size, num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader, test_loader

    def one_batch(self, data, model, meter, inference = False):
        data_in = self.pre_processing(data["inputs"])
        data["label"] = data["label"]
        outputs = model(data_in).squeeze()
        loss = self.criterion(outputs, data["label"])
        if not inference:
            meter.update_loss(loss.detach().cpu().numpy())
            meter.update_outputs(outputs.detach().cpu().numpy(), data["label"].cpu().numpy())
        else: 
            meter.add(data["pt_name"], data["label"], data["channel_name"], data["start_end"], outputs, loss)
        return loss.mean()

    def train(self, train_loader, valid_loader, checkpoint_folder, model = None):
        if model is None :
            model = self.__initialize_model()
        optimizer =  self.__initialize_optimizer(model)
        since = time.time()
        best_loss = 1e10
        best_model = None   
        for epoch in tqdm(range(self.num_epochs), disable=self.verbose):
            self.pre_processing.enable_random_shift()
            M = 1
            model.train()
            meter = Meter()
            for _, (pt_names, feature, label, channel_name, start_end) in enumerate(train_loader, 0):
                batch = pack_batch(pt_names, feature, label, channel_name, start_end, self.device)
                optimizer.zero_grad()
                loss = self.one_batch(batch, model, meter)
                loss.backward()
                optimizer.step()
            loss, acc = meter.loss(), meter.accuracy()
            if self.verbose:
                print("Epoch %d/%d, train loss: %.4f, train acc: %.4f" % (epoch, self.num_epochs, loss, acc))
                print("-" * 10)
            # Validation
            if epoch % M == 0 and epoch != 0:
                v_loss, v_acc  = self.validate(valid_loader, model, fn = None)
                best_loss, best_model = pick_best_model(model, best_model ,epoch, v_loss, best_loss, checkpoint_folder, model_name="best", preprocessing = self.pre_processing, save = self.save_checkpoint, verbose = self.verbose)
        time_elapsed = time.time() - since
        print(f"Training complete after {epoch} epochs in {int(time_elapsed // 60)} m {int(time_elapsed % 60)} s")
        return best_model
    
    def validate(self, loader, model, fn = None):
        start = time.time()
        meter = Meter("v_spike")
        model.eval()
        self.pre_processing.disable_random_shift()
        for _, (pt_names, feature, label, channel_name, start_end) in enumerate(loader, 0):
            with torch.no_grad():
                batch = pack_batch(pt_names, feature, label, channel_name, start_end, self.device)
                loss = self.one_batch(batch, model, meter, fn is not None)
        acc, loss = meter.accuracy(), meter.loss()
        if fn is not None and self.save_checkpoint:
            if self.verbose:
                print("--------------Testing---------------------")
            meter.dump_csv(os.path.join(fn, "test.csv"))
        if self.verbose:
            print("Validation: Time %.3f, loss: %.3f, acc: %.3f" % (time.time() - start, loss, acc))
        return loss, acc


if __name__ == "__main__":
    device = sys.argv[1]
    from src.args import args
    print(args)
    args['device'] = device
    args["data_dir"] = "data_training/artifact_data"   # data dir  
    args["augmentation_arg"]["selected_window_size_ms"] = 285       # window size in ms the actual window size is 2*selected_window_size_ms
    args["augmentation_arg"]["selected_freq_range_hz"] = [10, 300]  # frequency range in Hz
    args["augmentation_arg"]["random_shift_ms"] = 50                # random shift in ms in augmentation
    # get last folder name from data_dir   
    model_suffix = args["data_dir"].split("/")[-1]
    suffix = f"{model_suffix}_win{args['augmentation_arg']['selected_window_size_ms']}_freq{args['augmentation_arg']['selected_freq_range_hz'][0]}_{args['augmentation_arg']['selected_freq_range_hz'][1]}_shift{args['augmentation_arg']['random_shift_ms']}" 
    args["res_dir"] = os.path.join(args["res_dir"], suffix)
    trainer = Trainer(args, verbose=False)
    trainer.kfold_crossvalidation(5)

     