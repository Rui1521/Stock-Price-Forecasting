# Time Series
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils import data
import itertools
import random
import pandas as pd
import numpy as np
import os
import time
from itertools import combinations 
from torch.autograd import Variable
from torch.utils import data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")
from data_prep import train_valid_test

# read data
## M, AMZN, AAPL, SBUX, BAC
# 2010~2014, 2015, 2016
#frame = pd.read_csv("Data.csv")[['date', 'symbol', 'open', 'close', 'low', 'high', 'volume']]
#frame = frame.sort_values('date').reset_index(drop=True)


#RNN_lstms
def RMSE(pred, true, std):
    return round(np.sqrt(np.mean((pred[1:]*std - true[:-1]*std)**2)),5)

class Dataset(data.Dataset):
    def __init__(self, data_tensor, input_steps):
        self.X = data_tensor[:, :-1,:]
        self.Y = data_tensor[:, -1:,0]
        self.input_steps = input_steps
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x = self.X[index][-self.input_steps:, :]
        y = self.Y[index]
        return x, y

class LSTMs(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout_rate, device):
        super(LSTMs, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers = self.num_layers, 
                            dropout = dropout_rate, batch_first = True).to(device)
        
        self.output_layer = nn.Linear(self.hidden_dim, 1)
         
    def forward(self, x):
        output, _ = self.lstm(x) 
        output = self.output_layer(output[:,-1:,:].float())
        return output 
    
def run_epoch_train(model, data_generator, model_optimizer, criterion):
    preds = []
    trues = []
    for x, y in data_generator:
        # The input shape for nn.conv1d should sequence_length * batch_size * #features
        input_tensor, target_tensor = x.to(device).float(), y.to(device).float()
        model_optimizer.zero_grad()
        loss = 0
        output = model(input_tensor).reshape(target_tensor.shape)
        trues.append(target_tensor.cpu().detach().numpy())
        preds.append(output.cpu().detach().numpy())
        loss = criterion(output, target_tensor)
        loss.backward()
        model_optimizer.step()
    preds = np.concatenate(preds, axis = 0).squeeze(-1)
    trues = np.concatenate(trues, axis = 0).squeeze(-1)
    return preds, trues
#round(np.sqrt(np.mean(MSE)), 5)
 

def run_epoch_eval(model, data_generator, criterion, return_pred = False):
    with torch.no_grad():
        MSE = []
        preds = []
        trues = []
        for x, y in data_generator:
            input_tensor, target_tensor = x.to(device).float(), y.to(device).float()
            loss = 0
            output = model(input_tensor).reshape(target_tensor.shape)        
            #loss = criterion(output, target_tensor)
            #MSE.append(loss.item())
            preds.append(output.cpu().detach().numpy())
            trues.append(target_tensor.cpu().detach().numpy())
    preds = np.concatenate(preds, axis = 0).squeeze(-1)
    trues = np.concatenate(trues, axis = 0).squeeze(-1)
    return preds,trues

def train_model(model, train_set, valid_set, test_set, input_steps, learning_rate, batch_size = 50):
    # Initialize the model and define optimizer, learning rate decay and criterion
    optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 5, gamma=0.8)
    criterion = nn.MSELoss()
    
    # Split dataset into training set, validation set and test set.
    train_rmse, train_set = [], Dataset(train_set, input_steps)
    valid_rmse, valid_set = [], Dataset(valid_set, input_steps)
    test_rmse, test_set = [], Dataset(test_set, input_steps)
    best_model = 0
    best_preds = 0
    min_valid_loss = 1000
    
    
    for i in range(200):   
       #start = time.time()
        scheduler.step()
        train_generator = data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
        valid_generator = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False)
        test_generator = data.DataLoader(test_set, batch_size = batch_size, shuffle = False)

        model.train()
        train_preds, train_trues = run_epoch_train(model, train_generator, optimizer, criterion)
        train_rmse.append(RMSE(train_preds, train_trues, stds[0]))
        
        model.eval()
        valid_preds, valid_trues = run_epoch_eval(model, valid_generator, optimizer, criterion)
        test_preds, test_trues = run_epoch_eval(model, test_generator, optimizer, criterion)
        valid_rmse.append(RMSE(valid_preds, valid_trues, stds[0]))
        test_rmse.append(RMSE(test_preds, test_trues, stds[0]))
      
        if valid_rmse[-1] < min_valid_loss:
            min_valid_loss = valid_rmse[-1]
            min_test_loss = test_rmse[-1]
            best_model = model
            best_preds = test_preds
            
        if (len(train_rmse) > 20 and np.mean(valid_rmse[-5:]) >= np.mean(valid_rmse[-10:-5])):
            break
            
       # end = time.time()   
        """
        print(("Epoch %d:"%(i+1)), 
              ("Train_Loss: %f; "%train_rmse[-1]), 
              ("Valid_Loss: %f; "%valid_rmse[-1]),
              ("Test_Loss: %f; "%test_rmse[-1]),
              ("min_valid_loss: %f; "%min_valid_loss), 
              ("Time: %f; "%round(end - start,5)))
        """
    return best_model, (train_rmse, valid_rmse, test_rmse), min_valid_loss, min_test_loss, best_preds

# Random Paramter Search
features = ['open', 'low', 'high', 'volume']
features = list(combinations(features, 2)) + list(combinations(features, 1))
param_grid = {
    'learning rate': [0.01, 0.001, 0.0001],
    'dropout_rate': list(np.linspace(0.2, 0.8, 4)),
    'num_layers': [1],
    'hidden_dim': list(range(64, 256+64, 64)),
    "features": [[]] + list(map(list, features)) 
}
com = 1
for x in param_grid.values():
    com *= len(x)
# Only use 20 percent of total number of combinations
max_evals = int(com*0.1)

for company in ["M", "AAPL", "BAC"]:
    for input_steps in list(range(2, 22, 2)):
        output_steps = 1
        output_dim = 1

        best_params = []
        best_preds = []
        min_loss_all = 1000
        for i in range(0, max_evals):
            random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
            learning_rate = random_params["learning rate"]
            dropout_rate = random_params["dropout_rate"]
            num_layers = random_params["num_layers"]
            hidden_dim = random_params["hidden_dim"]
            feats = random_params["features"]
            input_dim = len(feats) + 1
            
            train_set, valid_set, test_set, (avgs, stds), train_tensor, pred_set = train_valid_test(frame, company, 31, feats, True)
            
            torch.cuda.empty_cache()
            model = LSTMs(input_dim, output_dim, hidden_dim, num_layers, dropout_rate, device).to(device)
            best_model, loss, min_valid_loss, min_test_loss, preds = train_model(model, train_set, valid_set, test_set, 
                                                             input_steps, learning_rate, batch_size = 50) 

            if min_valid_loss < min_loss_all: 
                min_loss_all = min_valid_loss 
                best_params = random_params
                best_model = model
                best_preds = preds

        print(company, input_steps, best_params, min_valid_loss, min_test_loss)
        