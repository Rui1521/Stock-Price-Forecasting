
import pandas as pd
import numpy as np
import os
import time
from itertools import combinations 
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from data_prep import train_valid_test, RMSE
import warnings
warnings.filterwarnings("ignore")



frame = pd.read_csv("Data.csv")[['date', 'symbol', 'open', 'close', 'low', 'high', 'volume']]
frame = frame.sort_values('date').reset_index(drop=True)
features = ['open', 'low', 'high', 'volume']
features = list(combinations(features, 1)) + list(combinations(features, 2)) + list(combinations(features, 3)) + list(combinations(features, 4))
features = list(map(list, features)) 

losses = {}
for company in ["M", "AAPL", "BAC"]:
    for lags in list(range(2,22,2)):
        min_valid_loss = 1000
        best_feats = []
        for feats in features:
            #print(feats)
            train_set, valid_set, test_set, (avgs, stds), train_tensor, pred_set = train_valid_test(frame, company, 31, feats)
            train_tensor_diff =  np.diff(train_tensor,axis = 0)
            
            train_set_diff, valid_set_diff, test_set_diff = np.diff(train_set,axis = 1), np.diff(valid_set,axis = 1), np.diff(test_set,axis = 1)
            model = VAR(train_tensor_diff)
            results = model.fit(lags)
            
            train_diff_x, train_y = train_set_diff[:,-1-lags:-1,:], train_set[:,-1,0]
            valid_diff_x, valid_y = valid_set_diff[:,-1-lags:-1,:], valid_set[:,-1,0]
            test_diff_x, test_y = test_set_diff[:,-1-lags:-1,:], test_set[:,-1,0]
            
            
            pred_train = train_set[:,-2,0] + np.array([results.forecast(train_diff_x[i], 1)[0][0] for i in range(train_diff_x.shape[0])])
            pred_valid = valid_set[:,-2,0] + np.array([results.forecast(valid_diff_x[i], 1)[0][0] for i in range(valid_diff_x.shape[0])])
            pred_test = test_set[:,-2,0] + np.array([results.forecast(test_diff_x[i], 1)[0][0] for i in range(test_diff_x.shape[0])])
         
            rmse_train = RMSE(pred_train, train_y, stds[0])
            rmse_valid = RMSE(pred_valid, valid_y, stds[0])
            rmse_test = RMSE(pred_test, test_y, stds[0])
            
            if rmse_valid < min_valid_loss:
                min_valid_loss = rmse_valid
                loss = [rmse_train, rmse_valid, rmse_test]
                best_feats = feats
    
        print(company, lags, loss, ['close'] + best_feats)

        