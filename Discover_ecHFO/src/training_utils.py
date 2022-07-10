import torch
import torch.nn as nn
import torch.optim as optim
from random import random
import random
import numpy as np

import os, time, copy,sys

def pick_best_model_acc(model, best_model ,epoch, v_acc, best_acc, checkpoint_folder, model_name="a"):
    save_checkpoint({'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            },
            filename= os.path.join(checkpoint_folder,f'model_{model_name}.pth'))
    if v_acc > best_acc:
        best_acc = v_acc 
        best_model =copy.deepcopy(model)
        print(f"best_model of {model_name} in epoch ", epoch +1)

        save_checkpoint({'epoch': epoch + 1,
                'state_dict': best_model.state_dict(),
                },
                filename= os.path.join(checkpoint_folder,f'model_s.pth'))
    return best_acc, best_model

def save_checkpoint(state, is_best=0, filename='models/checkpoint.pth.tar'):
    torch.save(state, filename)


def pick_best_model(model, best_model ,epoch, v_loss, best_loss, checkpoint_folder, model_name="a"):
    if v_loss < best_loss:
        best_loss = v_loss 
        best_model =copy.deepcopy(model)
        print(f"best_model of {model_name} in epoch ", epoch +1)
        save_checkpoint({'epoch': epoch + 1,
                'state_dict': best_model.state_dict(),
                },
                filename= os.path.join(checkpoint_folder, f'model_{model_name}.pth'))
    return best_loss, best_model
