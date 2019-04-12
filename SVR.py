import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from itertools import combinations 
import warnings
from data_prep import train_valid_test
warnings.filterwarnings("ignore")


## M, AMZN, AAPL, SBUX, BAC
# 2010~2014, 2015, 2016
frame = pd.read_csv("Data.csv")[['date', 'symbol', 'open', 'close', 'low', 'high', 'volume']]
frame = frame.sort_values('date').reset_index(drop=True)

features = ['open', 'low', 'high', 'volume']
features = list(combinations(features, 2)) + list(combinations(features, 1)) + list(combinations(features, 3)) + list(combinations(features, 4))
features = [[]] + list(map(list, features))

def RMSE(pred, true, std):
    return round(np.sqrt(np.mean((pred*std - true*std)**2)),5)

for company in ["M", "AAPL", "BAC"]:
    for lags in list(range(2,22,2)):  
        min_valid_loss = 1000
        best_params = []
        loss = []
        for feats in features:
            train_set, valid_set, test_set, (avgs, stds), train_tensor, pred_set = train_valid_test(frame, company, 31, feats)
            train_x, train_y = train_set[:,-1-lags:-1,:].reshape(-1,lags*(len(feats)+1)), np.concatenate(train_set[:,-1:,0])
            valid_x, valid_y = valid_set[:,-1-lags:-1,:].reshape(-1,lags*(len(feats)+1)), np.concatenate(valid_set[:,-1:,0])
            test_x, test_y = test_set[:,-1-lags:-1,:].reshape(-1,lags*(len(feats)+1)), np.concatenate(test_set[:,-1:,0])
            for C in [0.1,0.5,1,5,10,50,100]:
                for epsilon in [0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 10, 15]:
                    clf = SVR(C = C, epsilon = epsilon)
                    clf.fit(train_x, train_y) 
                    pred_valid = clf.predict(valid_x)
                    rmse_valid = RMSE(pred_valid, valid_y, stds[0])
                
        
                    if rmse_valid < min_valid_loss:
                        min_valid_loss = rmse_valid
                        
                        pred_train = clf.predict(train_x)
                        pred_test = clf.predict(test_x)
                        
                        rmse_valid = RMSE(pred_valid[1:], valid_y[:-1], stds[0])
                        rmse_train = RMSE(pred_train[1:], train_y[:-1], stds[0])
                        rmse_test = RMSE(pred_test[1:], test_y[:-1], stds[0])
                        
                        loss = (rmse_train, rmse_valid, rmse_test)
                        best_params = [feats, C, epsilon]
                        pred_test = clf.predict(valid_x)
    
        print(company, lags, loss, best_params)
        
        