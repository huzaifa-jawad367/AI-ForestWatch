import os
import pickle
import torch
import numpy as np

pkl_dir = 'Data/train/training_2015_pickled_data'

for fname in os.listdir(pkl_dir):
    if fname.endswith('.pkl'):
        fpath = os.path.join(pkl_dir, fname)
        with open(fpath, 'rb') as f:
            tup = pickle.load(f, encoding='latin1')
        tensor2 = tup[1]
        unique_vals = np.unique(tensor2)
        print(f"{fname}: unique elements in 2nd tensor: {unique_vals.tolist()}")
