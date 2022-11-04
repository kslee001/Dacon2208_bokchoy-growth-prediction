# coding: utf-8
import sys
import os
sys.path.append(os.getcwd())
# sys.path.append("./home/gyuseonglee/anaconda/envs/tch/lib/")


import pandas as pd
import numpy as np
from datetime import datetime as dt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.nn.utils import clip_grad_norm_ as clip_grad
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')


import columns # contains columns of data (origin, target)


def get_time():
    now = dt.now()
    cur_year = str(now.year)[-2:]
    cur_month = now.month
    cur_month = str(cur_month) if cur_month>=10 else '0'+str(cur_month)
    cur_date = now.day
    cur_date = str(cur_date) if cur_date >=10 else '0'+str(cur_date)
    return cur_year + cur_month + cur_date

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def preprocessing(input_list:list, target_list:list, infer:bool=False):     
    if not infer:
        # set target columns
        origin_columns = columns.origin_columns
        target_columns = columns.target_columns

        processed_X = []
        processed_Y = []
        for x,y in zip(input_list, target_list):
            curx = pd.read_csv(x)
            curx.columns = origin_columns
            curx = curx[target_columns].fillna(0).values
            x_len = len(curx)//1440
            x_temp = []
            for idx in range(x_len):
                x_temp.append(curx[1440*idx : 1440*(idx+1)])
            x_temp = torch.Tensor(x_temp)
            processed_X.append(x_temp)
            y_temp = torch.Tensor(pd.read_csv(y)["rate"].fillna(0).values)
            y_temp = y_temp.reshape(y_temp.size()[0], 1)
            processed_Y.append(y_temp)
        processed_X = torch.vstack(processed_X)
        processed_Y = torch.vstack(processed_Y)

        return processed_X, processed_Y
    
    else:
        # set target columns
        origin_columns = columns.origin_columns
        target_columns = columns.target_columns

        inputs  = []
        targets = []
        for idx in range(6):
            temp = []
            x = pd.read_csv(input_list[idx]).drop_duplicates().reset_index()
            del x["index"]
            x.columns = origin_columns
            x = x[target_columns].fillna(0).values
            if idx == 3:
                x = x[4320:]
            y = pd.read_csv(target_list[idx])
            
            x_len = len(x)//1440
            for i in range(x_len):
                temp.append(x[1440*i : 1440*(i+1)])
            temp = torch.Tensor(temp)
            inputs.append(temp)
            targets.append(y)

        return inputs, targets


def load_batch(X, Y, batch_size, shuffle=True):
    if shuffle:
        permutation = np.random.permutation(X.shape[0])
        X = X[permutation, :]
        Y = Y[permutation, :]
    num_steps = int(X.shape[0])//batch_size
    step = 0
    while step<num_steps:
        X_batch = X[batch_size*step:batch_size*(step+1)]
        Y_batch = Y[batch_size*step:batch_size*(step+1)]
        step+=1
        yield X_batch, Y_batch

        
def valid(model, device, optimizer, criterion, batch_size, X_val, Y_val):
    # mode change
    model.eval()
    val_loss = []
    num = 0
    # avoid unnecessary calculations
    with torch.no_grad():
        for X_batch, Y_batch in load_batch(X_val, Y_val, batch_size):
            num += batch_size
            # forward
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            yhat = model(X_batch)
                
            # loss
            loss = criterion(yhat, Y_batch)
            
            # save loss values
            val_loss.append(loss.item())
            
    return np.sum(val_loss)/num


def train(model, device, criterion, optimizer, scheduler, clip, X_train, Y_train, X_val, Y_val, lr, n_epochs, batch_size, max_norm):
    model.to(device)
    best_loss  = 999999999999
    best_model = None
    
    for epoch in range(1, n_epochs+1):
        num = 0
        train_loss = []
        
        # mode change
        model.train()
        for X_batch, Y_batch in load_batch(X_train, Y_train, batch_size):
            num+= batch_size
            
            optimizer.zero_grad()

            # forward
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)            
            yhat = model(X_batch)
            
            # loss
            loss = criterion(yhat, Y_batch)
            
            # backward
            loss.backward()
            if clip :
                clip_grad(model.parameters(), max_norm)
            optimizer.step()
            
            # save loss values
            train_loss.append(loss.item())
            
        val_loss = valid(model, device, optimizer, criterion, batch_size, X_val, Y_val)
        print(f'Train Loss : [{np.sum(train_loss)/num:.5f}] Valid Loss : [{val_loss:.5f}]')
        
        if scheduler is not None:
            if epoch > 25:
                scheduler.step()
            
        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model
            print(" -- best model found -- ")
    return best_model






## deprecated
def preprocessing__legacy(X_input, Y_input, X_container, Y_container):     
    origin_columns = [
        '시간', 
        '내부온도관측치', '내부습도관측치', 
        'CO2관측치', 'EC관측치', 
        '외부온도관측치', '외부습도관측치',
        '펌프상태', '펌프작동남은시간', 
        '최근분무량', '일간누적분무량', 
        '냉방상태', '냉방작동남은시간', 
        '난방상태', '난방작동남은시간', 
        '내부유동팬상태', '내부유동팬작동남은시간', 
        '외부환기팬상태', '외부환기팬작동남은시간',
        '화이트 LED상태', '화이트 LED작동남은시간', '화이트 LED동작강도', 
        '레드 LED상태', '레드 LED작동남은시간', '레드 LED동작강도', 
        '블루 LED상태', '블루 LED작동남은시간', '블루 LED동작강도', 
        '카메라상태', 
        '냉방온도', '난방온도', '기준온도', 
        '난방부하', '냉방부하', 
        '총추정광량', '백색광추정광량', '적색광추정광량', '청색광추정광량']
    target_columns = [
        # '시간', 
        '내부온도관측치', '내부습도관측치', 
        'CO2관측치', 'EC관측치', 
        '외부온도관측치', '외부습도관측치',
        # '펌프상태', '펌프작동남은시간', 
        '최근분무량', '일간누적분무량', 
        # '냉방상태', '냉방작동남은시간', 
        # '난방상태', '난방작동남은시간', 
        # '내부유동팬상태', '내부유동팬작동남은시간', 
        # '외부환기팬상태', '외부환기팬작동남은시간',
        '화이트 LED동작강도', #'화이트 LED상태', '화이트 LED작동남은시간', 
        '레드 LED동작강도', #'레드 LED상태', '레드 LED작동남은시간', 
        '블루 LED동작강도', #'블루 LED상태', '블루 LED작동남은시간',  
        # '카메라상태', 
        '냉방온도', '난방온도', '기준온도', 
        '난방부하', '냉방부하',  
        '백색광추정광량', '적색광추정광량', '청색광추정광량',#'총추정광량'
        '월' # will be generated in proprecessing
        ]


    for x,y in zip(X_input, Y_input):
        curx = pd.read_csv(x)
        curx.columns = origin_columns
        curx["월"] = curx["시간"].str.split("-", expand = True)[1].astype(int)
        curx["월"] = np.where(
            curx["월"]==12, 0, curx["월"] 
        )
        curx = curx[target_columns].fillna(0).values
        x_len = len(curx)//1440
        x_temp = []
        for idx in range(x_len):
            x_temp.append(curx[1440*idx : 1440*(idx+1)])
        x_temp = torch.Tensor(x_temp)
        X_container.append(x_temp)
        y_temp = torch.Tensor(pd.read_csv(y)["rate"].fillna(0).values)
        y_temp = y_temp.reshape(y_temp.size()[0], 1)
        Y_container.append(y_temp)