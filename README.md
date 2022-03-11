## Classification of high frequency oscillations using deep learning in pediatric chronic intracranial electroencephalogram

Source Code of paper

[Refining epileptogenic high-frequency oscillations using deep learning: A novel reverse engineering approach](https://www.biorxiv.org/content/10.1101/2021.08.31.458385v1)

Authors: Yipeng Zhang, Qiujing Lu, Tonmoy Monsoor, Shaun A. Hussain, Joe X Qiao, Noriko Salamon, Aria Fallah, Myung Shin Sim, Eishi Asano, Raman Sankar, Richard J. Staba, Anatol Bragin, Jerome Engel Jr., William Speier, Vwani Roychowdhury, and \*Hiroki Nariai

---

## Requirements

All of the code including Matlab and Python is tested under Ubuntu 18.04

The python code is tested under python 3.7.7 and torch 1.6.0

To install all of the required packages

```
conda create HFO-Classification
conda activate HFO-Classification
pip install -r requirements.txt
```

---

## How to classify your HFO into three types:

1.  Install the required packages following the instructions in Requirements section
2.  Prepare datset: follow Step 1,2 in Data pre-processing section. After Step 2, you should have following data hirerachy:

```
data_classification
│
└───patient_name1
│   │   data.npz
│
└───patient_name2
│   │   data.npz
```

3.  Prepare model: have your train model ready for artifacts, HFO-w-skipe and epileptogenic HFOs in ckpt folder and name them as

```
artifacts.pth
spikes.pth
eHFOs.pth
```

4. modify the fields in config/config_inference_HFO.ini

- **work_dir** : the root dir of this repo
- **data_dir** : the path for data_classification
- **res_dir** : the location you want to dump the result
- **model_dir** :the ckpt dir
- **device** : which device you want to use

5. Run the code below and the results will be dumped into res_dir folder as CSV

```
nohup python HFO_classification.py -c config/config_inference_HFO.ini > logs/log_inference &
```

---

## Data Pre-Processing

Extract HFO from original .edf file and convert the detected HFO information as well as its features to pytorch friendly dataset

### Input data format

The raw data hierarchy:

```
data_folder
│
└───patient_name1
│   │   data1.edf
│
└───patient_name2 (example for patient with multiple edfs)
    │   data1.edf
    │   data2.edf
```

### step 1: HFO detection

1.  Specify input_data_dir as the path of **your edf datafolder** in RIPPLELAB_API/run_HFO_detector.m line 6
2.  Run RIPPLELAB_API/run_HFO_detector.m, detected HFO result will be saved into **your edf datafolder** with filename endding with **".rhfe"**

### step 2: Network input feature preparation (For inference)

1. Specify **your edf datafolder** in config/config_preprocessing.ini in **datain_90** field
2. Run the following command and the **data.npz** will be created in each patient folder

```
python data_preprocessing/create_data_90min/get_spectrum.py -c config/config_preprocessing.ini
```

### step 3: Generate your TRAINING data (Only use the following steps if you want to retrain the networks)

Two tasks can be done using this framework, both of them requires

- Classify artifacts, HFO-with-spike and HFO-without-spike using annotated HFOs
- Discover eHFOs using trained artifacts classfier and channel-wise clinical information (soz, resected)

#### Classify artifacts, HFO-with-spike and HFO-without-spike

1. It require three **.rhfe** files in each patient folder, make sure these three files are in each patient folder
   ```
   {X}\_original.rhfe: raw HFO detection
   {X}-verified_STE.rhfe: annotated HFO without artifacts (from expert)
   {X}-HFO-spike_STE.rhfe: annotated HFO-with-spike (from expert)
   ```
2. specify the **datain_10** and **dataout_10** in config/config_preprocessing.ini
3. in the terminal run the following command

```
python data_preprocessing/create_data_10min/get_spectrum.py -c config/config_preprocessing.ini
```

```
python data_preprocessing/create_data_10min/generate_annoted_label.py -c config/config_preprocessing.ini
```

#### Discover eHFOs

It require one **.rhfe** file in each patient folder and a csv with channels' remove and soz information
a template file is located in data_preprocessing/Channel_Status_08162020.xlsx

- specify the **datain_90**, **dataout_90**, and **channel_annotation** in config/config_preprocessing.ini
- in the terminal run the following command

```
python data_preprocessing/create_data_90min/get_spectrum.py -c config/config_preprocessing.ini
```

**The labels for the training will be generated based on the artifacts model, we will go through it later**

---

## Train Models

- Specify all patient names in patient_info.py. The names should exactly match with those patient_nameX in **your edf datafolder**
- Specify seizure-free patient names. The names should be a subset of all patient names
- You may use `os.listdir(your edf datafolder)` to get all patient names from your edf datafolder

### Classify artifacts, HFO-with-spike and HFO-without-spike

#### kfold training

- In config/config_10min_all_patient.ini, Specify **working_dir** (the "root" path) **data_dir** (same as dataout_10 in data preprocessing step3) and **res_dir**
- In the termimal run the following command

```
nohup python train.py -c config/config_10min_all_patient.ini > logs/log_train_10min_all_patient &
```

#### patient-wise cross validation

- In config/config_10min_patientwise_elimination.ini, specify **working_dir** (the "root" path) **data_dir** (same as dataout_10 in data preprocessing step3) and **res_dir**
- In the termimal run the following command

```bash
nohup python train.py -c config/config_10min_patientwise_elimination.ini > logs/log_train_10min_patientwise_elimination &
```

### Discover eHFOs

#### Step 1: Using one of trained artifacts classifier to classify 90 min HFO data

- Specify the required fields in config/config_10min_inference_90min.ini
- In the termimal run the following command

```bash
nohup python inference.py -c config/config_10min_inference_90min.ini  > logs/log_10min_inference_90min &
```

#### Step 2: Generate artifact label for 90 min HFO data

- Specify the required fields in config/config_10min_inference_90min.ini
- In the termimal run the following command

```bash
nohup python inference.py -c config/config_10min_inference_90min.ini  > logs/log_10min_inference_90min &
```

- Specify **inference_res** (same as **res_dir** in previous step) in config/config_preprocessing.ini

```bash
python data_preprocessing/create_data_90min/generate_remove_label.py -c config/config_preprocessing.ini
```

#### Step 3: Train eHFO model using kfold training

- Specify the required fields in config/config_90min_all_patient.ini
- In the termimal run the following command

```bash
nohup python train_reverse.py -c config/config_90min_all_patient.ini > logs/log_train_reverse_all_patients &
```

#### Step 4 (alternative to Step 3): Train eHFO model using patient-wise cross validation

- Specify the required fields in config/config_90min_patientwise_elimination.ini
- In the termimal run the following command

```bash
nohup python train_reverse.py -c config/config_90min_patientwise_elimination.ini > logs/log_train_90min_patientwise_elimination &
```

#### Step 5: Inference trained model

```bash
nohup python inference_reverse.py -c config/config_90min_inference_90min.ini > logs/log_90min_inference_90min &
```

```bash
nohup python inference_reverse.py -c config/config_90min_inference_10min.ini > logs/log_90min_inference_10min &
```

## Contract:
You can email zyp5511@g.ucla.edu if you have any question :)
