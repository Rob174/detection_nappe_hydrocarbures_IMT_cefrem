import numpy as np


def moving_mean(x,window):
    new_values = []
    for i in range(len(x)-window):
        new_values.append(np.mean(x[i:min(i+window,len(x))]))
    return new_values