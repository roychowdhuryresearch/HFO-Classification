data_args = {
    "image_size": 224,
    "time_window_ms": 1000,
    "freq_min_hz": 10,
    "freq_max_hz": 500,
    "resample": 2000,
    "n_jobs": 8,
    "n_feature": 1,
}

augmentation_arg = {
    "random_shift_ms": 45,
    "selected_window_size_ms": 285,
    "selected_freq_range_hz": [10, 290],
}

args = {
    'work_dir': '.',
    'data_dir': '/mnt/SSD3/New_patient_90/training_data_10_python',   
    'res_dir': 'result/',
    'num_epochs': 30,
    'batch_size':128,
    'learning_rate': 0.0003,
    'seed': 0,
    'p_val': 0.2,
    'device': 'cuda:1',
    'augmentation_arg': augmentation_arg,
    "save_checkpoint": True,
}

