import pickle
import random
import numpy as np
import torch


def save_pkl(path, obj):
    with open(path, 'wb') as writer:
        pickle.dump(obj, writer, pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(path, 'rb') as reader:
        obj = pickle.load(reader)
    return obj


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
