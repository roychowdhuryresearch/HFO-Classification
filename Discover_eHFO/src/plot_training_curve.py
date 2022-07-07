import numpy as np
import matplotlib.pyplot as plt
import os 
from utilities import load_pickle

def draw_curve(a, save_folder, patient_name):
    plt.figure()
    plt.subplot(311)
    plt.plot(a["training_loss"], label='train_loss',color='blue')
    plt.plot(a["validation_loss"], label='val_loss',color='orange')
    plt.title("loss")
    plt.legend()
    plt.subplot(312)
    plt.plot(a["training_acc"], label='train_acc',color='blue')
    plt.plot(a["validation_acc"], label='vali_acc',color='orange')
    plt.plot(a["test_acc"], label='test_acc', color='red')
    plt.title("accuracy")
    plt.ylim([0.5, 1.0])
    plt.legend()
    plt.subplot(313)
    #plt.plot(a["training_f1"], label='train_f1')
    plt.plot(a["validation_f1"], label='vali_f1',color='orange')
    plt.plot(a["test_f1"], label='test_f1', color='red')
    plt.title("F1-score")
    plt.legend()
    plt.suptitle(patient_name)
    plt.savefig(os.path.join(save_folder, patient_name+".jpg"))
    plt.close()
if __name__ == "__main__":
    
    '''
    This file is to plot the training curve,

    path: in result of training folder,  there is a file called  training_curve_a.pkl and training_curve_s.pkl
          which contains train stats.  
    out_folder : the output folder to dump the plot   
    '''

    path = ""
    out_folder = ""
    for p_name in sorted(os.listdir(path)):
        stats = load_pickle(os.path.join(path, p_name,"training_curve_a.pkl"))
        draw_curve(stats,out_folder,p_name+"_artifacts")
        stats = load_pickle(os.path.join(path, p_name,"training_curve_s.pkl"))
        draw_curve(stats,out_folder,p_name+"_spike")