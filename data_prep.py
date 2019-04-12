# Time Series
from __future__ import unicode_literals, print_function, division
import torch
import numpy as np
import pandas as pd
import itertools
import random
import pandas as pd
import numpy as np
from torch.autograd import Variable
from torch.utils import data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def RMSE(pred, true, std):
    return round(np.sqrt(np.mean((pred*std - true*std)**2)),5)

def sub_seqs(tensor, length):
        seqs = []
        for i in range(0, len(tensor) - length):
            seqs.append(tensor[i:i+length])
        return np.array(seqs)

def train_valid_test(frame, company, length, features, torch_tensor = False):
    subframe = frame[frame.symbol.isin([company])]
    subframe = subframe.sort_values('date').reset_index(drop=True)
    date = np.array(subframe.date)

    train_pos = np.where(date < "2015")[0]
    valid_pos = np.where((date < "2016") & (date >= "2015"))[0]
    test_pos = np.where(date >= "2016")[0]
    pred_pos = np.array(range(valid_pos[0]-30, test_pos[-1]))

    train_tensor = subframe.iloc[train_pos,:].sort_values('date').reset_index(drop=True)[["close"] + features].values
    valid_tensor = subframe.iloc[valid_pos,:].sort_values('date').reset_index(drop=True)[["close"] + features].values
    test_tensor = subframe.iloc[test_pos,:].sort_values('date').reset_index(drop=True)[["close"] + features].values
    pred_tensor = subframe.iloc[pred_pos,:].sort_values('date').reset_index(drop=True)[["close"] + features].values
    
    train_set, valid_set, test_set, pred_set= sub_seqs(train_tensor, length),\
                                              sub_seqs(valid_tensor, length),\
                                              sub_seqs(test_tensor, length),\
                                              sub_seqs(pred_tensor, length)
    
    avgs, stds = [], []
    
    for i in range(train_set.shape[-1]): 
        avg = np.mean(train_set[:,:,i])
        std = np.std(train_set[:,:,i])
        
        train_set[:,:,i] = (train_set[:,:,i] - avg)/std
        valid_set[:,:,i] = (valid_set[:,:,i] - avg)/std
        test_set[:,:,i] = (test_set[:,:,i] - avg)/std
        
        train_tensor[:,i] = (train_tensor[:,i] - avg)/std
        pred_set[:,:,i] = (pred_set[:,:,i] - avg)/std
        
        avgs.append(avg)
        stds.append(std)
        
    if torch_tensor:
        train_set, valid_set, test_set, pred_set = torch.from_numpy(train_set).float(), \
                                                   torch.from_numpy(valid_set).float(), \
                                                   torch.from_numpy(test_set).float(),\
                                                   torch.from_numpy(pred_set).float()
   
    return train_set, valid_set, test_set, (avgs, stds), train_tensor, pred_set