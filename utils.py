import pickle
import random
import numpy as np
import torch
import os
import time


def save_pkl(path, obj):
    with open(path, 'wb') as writer:
        pickle.dump(obj, writer, pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(path, 'rb') as reader:
        obj = pickle.load(reader)
    return obj


def set_random_seed(seed=None):
    if seed is None:
        # Generate a new random seed, e.g., based on the current time
        seed = int(time.time()) + os.getpid()
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
