#!/usr/bin/env python
# coding: utf-8
## Two separate models for artifact classification and HFO with spike classification
###

import os, time, copy, sys

import torch
import torch.nn as nn
import torch.optim as optim
from random import random
import random
import numpy as np

from src.utilities import *
from src.dataloader_spike import create_split_loaders_overall, create_patient_fold
from src.model import NeuralCNN
from src.config import arg_parse90
from src.meter import TrainingMeter, Meter
from src.training_utils import *
from patient_info import seizure_free_patient_names


def validate(val_loader, model, criterion, computing_device, fn=None):
    start = time.time()
    meter_s = Meter("spike")
    model_s = model["spike"]
    model_s.eval()
    for mb_count, (image, waveform, intensity, label, info, start_end) in enumerate(
        val_loader, 0
    ):

        with torch.no_grad():
            a_, s_, train_s = create_sa_labels(
                image, waveform, intensity, label, info, start_end, computing_device
            )

            outputs_s = model_s(s_["inputs"]).squeeze()
            s_["label"] = s_["label"].squeeze()[:, 0]
            # print(outputs_s.shape)
            if outputs_s.dim() == 0:
                outputs_s = outputs_s.unsqueeze(0)
                s_["label"] = s_["label"].unsqueeze(0)

            loss_s = criterion(outputs_s, s_["label"])
            if not fn:
                meter_s.update_loss(loss_s.detach().cpu().numpy())
                meter_s.update_outputs(
                    outputs_s.detach().cpu().numpy(), s_["label"].cpu().numpy()
                )
            else:
                meter_s.add(
                    s_["spectrum"],
                    s_["label"].detach().cpu().numpy(),
                    s_["channel_name"],
                    s_["start_end"],
                    s_["intensity"],
                    s_["waveform"],
                    outputs_s.detach().cpu().numpy(),
                )
    acc_s = meter_s.accuracy()
    if fn is not None:
        loss_s = 0
        meter_s.dump_pickle(os.path.join(fn, "spikes.pkl"))
    else:
        loss_s = meter_s.loss()
    f1_s = meter_s.f1()
    print(
        "Inference: Time %.3f, loss_s: %.3f, accuracy_s: %.3f , f1_s: %0.3f"
        % (time.time() - start, loss_s, acc_s, f1_s)
    )

    return loss_s, acc_s, f1_s


def create_sa_labels(
    image, waveform, intensity, label, channel_name, start_end, computing_device
):
    channel_name = np.array(channel_name)
    label = label.squeeze().float()

    # print("select_index", select_index, select_index.shape)
    s_image = image
    # s_image_norm = normalize_img(torch.log(s_image + 1e-8))
    # s_image_norm = torch.log(s_image)
    s_image_norm = s_image
    inputs_s = (
        torch.stack([s_image_norm, waveform, intensity], dim=1, out=None)
        .to(computing_device)
        .float()
    )
    # inputs_s = torch.stack([intensity[select_index],intensity[select_index],intensity[select_index]], dim=1, out=None).to(computing_device).float()
    label_s = label.to(computing_device)
    # print(inputs_s.shape)
    s_ = {
        "inputs": expand_dim(inputs_s, 4),
        "spectrum": image,
        "label": expand_dim(label_s, 1).squeeze(),
        "intensity": intensity,
        "waveform": waveform,
        "channel_name": channel_name,
        "start_end": start_end,
    }

    return None, s_, True


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    computing_device,
    num_epochs_s=10,
    checkpoint_folder=None,
    weight=0.5,
):
    since = time.time()
    best_acc_s = 0
    if not os.path.exists(checkpoint_folder):
        os.mkdir(checkpoint_folder)

    optimizer_s = optimizer["spike"]
    model_s = model["spike"]

    best_model_s = None

    train_meter_s = TrainingMeter("spike")

    for epoch in range(num_epochs_s):
        M = 1
        print("-" * 10)
        epoch_loss = 0
        # Each epoch has a training and validation phase

        model_s.train()

        meter_s = Meter("spike")

        for _, (image, waveform, intensity, label, info, start_end) in enumerate(
            train_loader, 0
        ):
            a_, s_, train_s = create_sa_labels(
                image, waveform, intensity, label, info, start_end, computing_device
            )
            optimizer_s.zero_grad()
            outputs_s = model_s(s_["inputs"])
            outputs_s = outputs_s.squeeze(1)
            s_["label"] = s_["label"]
            loss_s = (1- weight* torch.logical_xor(s_["label"][:, 1].long(), s_["label"][:, 0].long())) * criterion(outputs_s, s_["label"][:, 0])
            meter_s.update_loss(loss_s.detach().cpu().numpy())
            meter_s.update_outputs(
                outputs_s.detach().cpu().numpy(), s_["label"][:, 0].cpu().numpy()
            )
            loss_s = torch.sum(loss_s) * 1.0 / len(outputs_s)
            loss_s.backward()
            optimizer_s.step()

        # Print the loss averaged over the last N mini-batches
        loss_s = meter_s.loss()
        acc_s = meter_s.accuracy()
        print("Epoch %d, loss_s: %.3f, accuracy_s: %.3f" % (epoch + 1, loss_s, acc_s))

        # Validation
        if epoch % M == 0 and epoch != 0:
            v_loss_s, v_acc_s, v_f1_s = validate(
                val_loader, {"spike": model_s}, criterion, computing_device
            )
            best_acc_s, best_model_s = pick_best_model_acc(
                model_s,
                best_model_s,
                epoch,
                v_acc_s,
                best_acc_s,
                checkpoint_folder,
                model_name="s",
            )
            # print("----test_-----")
            # t_loss_s, t_acc_s, t_f1_s = validate(test_loader,{ "spike": model_s}, criterion, computing_device)
            train_meter_s.add(acc_s, loss_s, v_loss_s, v_acc_s, 0, v_f1_s, 0, 0)
    print("Training complete after", epoch, "epochs")
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s ".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    train_meter_s.dump_pickle(os.path.join(checkpoint_folder, "training_curve_s.pkl"))
    return {"spike": best_model_s}


def pipeline(args, test_patient_name):

    model_spike = NeuralCNN(num_classes=2, freeze_layers=True, dropout_p=0)

    data_dir = args.data_dir
    res_dir = os.path.join(args.work_dir, args.res_dir)  #
    num_epochs_s = args.num_epochs_s  # Number of full passes through the dataset
    batch_size = args.batch_size  # Number of samples in each minibatch
    learning_rate_s = args.learning_rate_s
    seed = args.seed  # Seed the random number generator for reproducibility
    p_val = args.p_val  # Percent of the overall dataset to reserve for validation
    p_test = args.p_test  # Percent of the overall dataset to reserve for testing
    weight = args.weight

    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device(f"{args.device}")
        extras = {"num_workers": 1, "pin_memory": True}
        print("CUDA is supported")
    else:  # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")
    # return

    model_spike = model_spike.to(computing_device)
    model = {"spike": model_spike}

    print("Model on CUDA?", next(model_spike.parameters()).is_cuda)

    criterion = nn.BCELoss(reduction="none").to(computing_device)

    optimizer_spike = optim.Adam(
        filter(lambda p: p.requires_grad, model_spike.parameters()), lr=learning_rate_s
    )
    optimizer = {"spike": optimizer_spike}

    start_time = time.time()

    if args.all_patient:
        train_loader, val_loader, test_loader = create_split_loaders_overall(
            data_dir,
            -1,
            batch_size,
            seed=seed,
            p_val=p_val,
            p_test=p_test,
            shuffle=True,
            show_sample=False,
            extras={},
        )
        patient_name = "overall"
    else:
        train_loader, val_loader, test_loader = create_patient_fold(
            data_dir,
            test_patient_name,
            batch_size,
            p_val=p_val,
            shuffle=True,
            show_sample=False,
            extras={},
        )
        patient_name = test_patient_name
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f"Prepare dataset | Time: {epoch_mins}m {epoch_secs}s")

    print("patient_names is", patient_name)
    stats_folder = os.path.join(res_dir, patient_name)
    clean_folder(stats_folder)

    print("----------------Training----------------")
    model_trained = train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        computing_device,
        num_epochs_s=num_epochs_s,
        checkpoint_folder=stats_folder,
        weight=weight,
    )

    print("-----------------testing ----------------")
    print("patient_names is", patient_name)
    loss_s, acc_s, _ = validate(
        test_loader, model_trained, criterion, computing_device, fn=stats_folder
    )
    return loss_s, acc_s


if __name__ == "__main__":
    args = arg_parse90(sys.argv[1:])
    print(args)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    clean_folder(os.path.join(args.work_dir, args.res_dir))
    if args.all_patient:
        print("all")
        loss_s, acc_s = pipeline(args, None)
    else:
        for p_name in seizure_free_patient_names:
            loss_s, acc_s = pipeline(args, p_name)
