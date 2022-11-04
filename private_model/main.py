# coding: utf-8
import sys
import os
sys.path.append(os.getcwd())
# sys.path.append("./home/gyuseonglee/anaconda/envs/tch/lib/")

from datetime import datetime as dt
import glob
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import functions
import models


os.chdir("..")

if __name__ == "__main__":  
    '''
    0. Prepare data
    '''
    # get data list
    train_input_list  = sorted(glob.glob(os.getcwd() + "/train_input/*.csv"))
    train_target_list = sorted(glob.glob(os.getcwd() + "/train_target/*.csv"))
    
    # run preprocessing function
    processed_X, processed_Y = functions.preprocessing(train_input_list, train_target_list, infer=False)
    print("\n--data processed : shapes of processed X and Y")
    print(processed_X.shape)
    print(processed_Y.shape)

    # train-test(valid) split
    X_train, X_val, Y_train, Y_val = train_test_split(
        processed_X, processed_Y, test_size = 0.1, shuffle = False
    )
    print("\n--data splited")
    print("# of X_train : ", len(X_train))
    print("# of Y_train : ", len(Y_train))
    print("# of X_val : ", len(X_val))
    print("# of Y_val : ", len(Y_val))


    '''
    1. Set hyperparameters
    '''
    # model name : current YYMMDD_[cluster name]
    model_name = functions.get_time() + "_AB"
    model_states_name = model_name +"_states"

    # make directory for current model
    functions.make_directory(os.getcwd() + "/" + model_name + "_folder")

    # hyper parameters and data shapes
    lr = 0.0005
    n_epochs = 2
    max_norm = 10
    N = 64      # batch_size
    T = processed_X.shape[1]
    I = processed_X.shape[2]
    O = processed_Y.shape[1]  
    H = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # check parameters
    print("\n-- hyperparameters ")
    print(f"current device : {device}")
    print(f"lr             : {lr}")
    print(f"n_epochs       : {n_epochs}")
    print(f"max_norm       : {max_norm}")
    print(f"N (batch_size)  : {N}")
    print(f"T (time_step)   : {T}")
    print(f"I (input_dim)   : {I}")
    print(f"H (hidden_dim)  : {H}")
    print(f"O (output_dim)  : {O}")

    # empty cuda cache
    torch.cuda.empty_cache()


    '''
    2. Define model and addons
    '''
    model = models.BidirectionalEncoder(I, H, O)
    print(f"\n model : [ {model.name} ] \n")    

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr = lr,
        eps = 1e-08,
        weight_decay = 0.1
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    clip      = True


    '''
    3. Run model
    '''
    best_model = functions.train(
        model, 
        device, 
        criterion, 
        optimizer, 
        scheduler, 
        clip, 
        X_train, 
        Y_train, 
        X_val, 
        Y_val, 
        lr, 
        n_epochs, 
        N,
        max_norm
    )


    '''
    4. Save model and record results
    '''
    # curmodel = torch.load(model_name)
    curmodel = best_model

    # save model
    torch.save(
        curmodel, 
        os.getcwd() + "/" + model_name + "_folder/" + model_name
    )
    torch.save(
        curmodel.state_dict(),
        os.getcwd() + "/" + model_name + "_folder/" + model_states_name
    )

    # save outputs
    input_list  = sorted(glob.glob(os.getcwd() + "/test_input/*.csv"))
    target_list = sorted(glob.glob(os.getcwd() + "/test_target/*.csv"))
    inputs, targets = functions.preprocessing(input_list, target_list, infer=True)
    for i in range(len(inputs)):
        with torch.no_grad():
            output = curmodel.forward(inputs[i].to("cuda")).to("cpu").squeeze(1)
            cur_target = pd.read_csv(target_list[i])
            cur_target["rate"] = output
            PATH = os.getcwd() + "/" + model_name + f"_folder/TEST_0{i+1}.csv"
            cur_target.to_csv(PATH, index = False)

    print("\n-- All outputs are saved at : \n[ " + os.getcwd() + "/" + model_name + "_folder/ ]\n")
