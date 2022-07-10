## Training
```bash
python Trainer_echfo.py -c config/config_90min_all_patient_weights.ini
```

## Inference (handling both)
```bash
python inference_ecHFO.py -c config/config_90min_all_patient_weights.ini
```

## Inference (on 10min) # no args here
```bash
python inference_ecHFO10min.py
```

## Generate Bell shape
```bash
python bell_curve_generation.py
```

## Bell shape perturbation
```bash
python bell_perturb_ecHFO.py
python bell_perturb_ecHFO_both.py
```

## Bell shape perturbation (these 2 take long)
```bash
python bell_perturb_ecHFO.py
python bell_perturb_ecHFO_both.py
```

## visualiation 
```bash
bell_curve_occlusion.ipynb
```

## spike shape perturbation
```bash
python min_col_ecHFO.py
python min_col_ecHFO_both.py
```

## aggregate results
paper_scripts/min_col_perturb_data_generation.py
paper_scripts/bell_shape_perturb_data_generation.py
paper_scripts/spike_perturb_data_generation.py
## time domain analysis
python time_domain_analysis.py

## ecHFO results analysis
/mnt/SSD3/yipeng/HFO_Classification/visulization/ecHFO_results_analysis.ipynb

## label label_distribution
/mnt/SSD3/yipeng/HFO_Classification/visulization/label_distribution.ipynb

## generate waveform
python data_preprocessing/create_spike_template/get_waveform.py -c config/config_preprocessing.ini


## Training
```bash
python Trainer_echfo_f.py -c config/config_90min_all_patient_weights_f.ini
```
