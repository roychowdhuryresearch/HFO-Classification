## Characterizing physiological high-frequency oscillations using deep learning

please follow the data preprocessing steps in Discover_eHFO or end2end/DataPreprocessing sections. 

For label assignment for each HFO:
* 1 is Behaviour 
* 0 is Non-behaviour
* -1 is Both

### Model Training:

Please specify your path in *config/config_ecHFO_training.ini*


#### Training
```bash
python trainer.py -c config/config_ecHFO_training.ini
```

#### Inference (handling both)
```bash
python inference.py -c config/config_ecHFO_training.ini
```




