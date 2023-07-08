## step 0 dataset preparation
```
data/task_name
│
└───patient_name1
│   │   XX.edf
│   │   XX.edf
│   │   annotation.csv
│
└───patient_name2
│   │   XX.edf
│   │   annotation.csv
'       
'       
'
└───patient_nameN
    │   XX.edf
    │   annotation.csv
```


## Step 1 Feature extraction

1. replace the **data_folder** in ***feature_extraction.py*** with your own data folder. For example ``data/artifact_data``, the task_name is *artifact_data*

2. replace other parameters as you need

3. after running the following command, you will get a folder with the name same as the *{task_name}* in the ``data_training`` folder. The folder contains all the features extracted from the edf files. for example in this case, the folder name is ``artifact_data``.

```
python feature_extraction.py
```

## Step 2 Train model

1. replace the **data_folder** in trainer.py with the folder name you get in step 1. For example ``data_training/artifact_data`` 

2. replace other parameters as you need, these parameters must be smaller than the ones in step 1.

3. after running the following command, you will get a folder with the name as 
``{task_name}_win{selected_window_size_ms}_freq{selected_freq_range_hz}_{selected_freq_range_hz}_shift{random_shift_ms}`` in the ``result`` folder. The folder contains all the trained models in 5 folds cross-validation. All of the parameters, such as *selected_window_size_ms*, *selected_freq_range_hz*, *random_shift_ms* are the same as the ones you specified in  ***trainer.py***. For example, in this case, the folder name is *artifact_data_win285_freq10_300_shift50*. In this folder, there are 5 sub-folders, each of them containing a trained model named "model_best.tar", and "test.csv" which contains the prediction results on the test set.

4. You need to specify the GPU you want to use in the ***trainer.py***. For example, if you want to use the first GPU, you need to run the following command. If you want to use the second GPU, you need to change the "cuda:0" to "cuda:1".

```
python trainer.py cuda:0
```

## Step 3 prune model

1. The pruning process is based on the trained model in step 2. The pruning process will prune the model and save the pruned model in the same folder as the trained model. The pruned model will be named as "pruned_model.tar".
2. You need to specify which fold you want to prune in the ***pruneer.py***. For example, if you want to prune the model in the first fold, you need to make the *model_folder* in ***pruneer.py*** as *result/artifact_data_win285_freq10_300_shift50/ckpt/fold_0*.
3. Other parameters you do not need to change.

4. The GPU to use in this step is the same as the one in step 2. For example, if you want to use the first GPU, you need to run the following command. If you want to use the second GPU, you need to change the "cuda:0" to "cuda:1".

5. After running the following command, you will get a pruned model in the same folder as the trained model. The pruned model will be named "pruned_model.tar".

```
python pruneer.py cuda:0
```

