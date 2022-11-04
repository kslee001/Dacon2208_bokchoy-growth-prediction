'''
        1. Import libraries
'''
import sys
sys.path.append("./home/gyuseonglee/anaconda/envs/tch/lib/")
import os
import pandas as pd
import numpy as np
import random
import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.nn.utils import clip_grad_norm_ as clip_grad
# from tqdm.auto import tqdm as tq
import warnings
warnings.filterwarnings(action='ignore')


'''
        2. Define functions ( + loss function )
'''
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


def preprocessing(X_input, Y_input, X_container, Y_container):     
    for x,y in zip(X_input, Y_input):
        curx = pd.read_csv(x)
        curx.columns = origin_columns
        curx["월"] = curx["시간"].str.split("-", expand = True)[1].astype(int)
        curx["월"] = np.where(
            curx["월"]==12, 0, curx["월"] 
        )
        try:
            curx = curx[target_columns].fillna(0).values
        except:
            curx = curx[target_columns2].fillna(0).values
        x_len = len(curx)//1440
        x_temp = []
        for idx in range(x_len):
            x_temp.append(curx[1440*idx : 1440*(idx+1)])
        x_temp = torch.Tensor(x_temp)
        X_container.append(x_temp)
        y_temp = torch.Tensor(pd.read_csv(y)["rate"].fillna(0).values)
        y_temp = y_temp.reshape(y_temp.size()[0], 1)
        Y_container.append(y_temp)

        
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
            if epoch > 20:
                scheduler.step()
            
        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model
            print(" -- best model found -- ")
    return best_model


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, yhat, ygt):
        return torch.sqrt(self.mse(yhat, ygt))


'''
        3. Define layers (+ layer returning functions)
'''
class Residual_original(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))


class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return tensors[0] + self.dropout(self.norm(self.sublayer(*tensors)))
    
    
def feed_forward(dim_input, dim_feedforward) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.GELU(), # activation function of original model was relu 
        nn.Linear(dim_feedforward, dim_input),
)


def position_encoding(seq_len, dim_model, 
                      device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) -> torch.Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim // dim_model))
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


def scaled_dot_product_attention(query:torch.Tensor, key:torch.Tensor, value:torch.Tensor)->torch.Tensor:  # value -> value
    # temp : Q @ K.T   -> (N.T.D) @ (N,D,T)  (batch size만 그대로 두고, D,T 와 T, D를 )
    # Q @ K.T -> (N, T, T)
    QKT     = query.bmm(key.transpose(1, 2))  # bmm : batch matrix multiplication (X, O, O)-> O에 해당되는 dim에 대해서만 matmul 진행
    root_dk = query.size(-1)**0.5            # squared root of D
    softmax = f.softmax( QKT / root_dk, dim= -1 ) # softmax for "T of Key", not for "T of Query", so dim = -1 is right
                                                  # dim = -2 로 맞추면 Key 에 대한 쿼리 결과(세로축)으로 1을 합산하는 꼴임
    return softmax.bmm(value) # (N,T,T)@(N,T,D) -> (N, T, D)


class AttentionHead(nn.Module): # X = (N, D)  / Wq, Wk, Wv : (D, Q) or (D, K) / 일반적으로 K, Q는 같은 dimension 사용
    def __init__(self, input_dim:int, query_dim:int, key_dim:int):
        super().__init__()
        
        self.q_linear = nn.Linear(input_dim, query_dim) # generate Q 
        self.k_linear = nn.Linear(input_dim, key_dim) # generate K
        self.v_linaer = nn.Linear(input_dim, key_dim) # generate V


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # 인풋으로 들어온 query, key, value 텐서에 대해 linear forward를 진행하고
        # 그 과정을 통해 만들어진 Q, K, V를 그대로 scaled_dot_product_attention 에 forward 시킴        
        return scaled_dot_product_attention(
            self.q_linear(query), # query @ q_linear : (Xq : N, D) @ (Wq : D, Q) -> (N, Q)
            self.k_linear(key),   # key   @ k_linear : (Xk : N, D) @ (Wk : D, K) -> (N, K)
            self.v_linaer(value)) # value @ v_linear : (Xv : N, D) @ (Wv : D, K) -> (N, K)
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, input_dim: int, query_dim: int, key_dim: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(input_dim, query_dim, key_dim) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * key_dim, input_dim)  
        # num_heads 만큼 horizontally concat 되므로, 
        # multiheadAttention의 forward 결과 나오는 concated V의 dimension 은 numm_heads배 늘어난다. 
        # 즉, (N, T, K) * num_heads -> (N, T, num_heads * K)


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return self.linear(
            torch.cat([ head(query, key, value) for head in self.heads], dim=-1)
        )


class TransformerEncoderLayer(nn.Module):
    def __init__(self, num_heads, dim_model, dim_feedforward, dropout):
        super().__init__()
        dim_q = max(dim_model // num_heads, 1)
        dim_k = dim_q
        self.attention = MultiHeadAttention(
                num_heads = num_heads,
                input_dim = dim_model,
                query_dim = dim_q, 
                key_dim   = dim_k
        )
        self.residual_AT = Residual(
            sublayer  = self.attention,
            dimension = dim_model,
            dropout   = dropout
        )
        self.feed_forward = feed_forward(
            dim_input       = dim_model,
            dim_feedforward = dim_feedforward
        )
        self.residual_FF = Residual(
            sublayer = self.feed_forward,
            dimension = dim_model,
            dropout   = dropout,
        )

    ''' source shape : (N, T, D)'''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_AT(x, x, x)
        return self.residual_FF(x)


class TransformerEncoder(nn.Module):
    def __init__(self, 
                 num_layers, 
                 num_heads, 
                 dim_model, 
                 dim_feedforward, 
                 dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    num_heads = num_heads, 
                    dim_model = dim_model, 
                    dim_feedforward = dim_feedforward, 
                    dropout = dropout
                )
                for _ in range(num_layers)
            ]
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape
        if(x.ndim==2):
            x = x.reshape(1, x.size(0), x.size(1))
        N, T, D = x.shape
        
        # positional encoding
        x += position_encoding(T, D)
        for layer in self.layers:
            x = layer(x)

        return x


'''
        4. Data preprocessing
'''

origin_columns = ['시간', '내부온도관측치', '내부습도관측치', 'CO2관측치', 'EC관측치', '외부온도관측치', '외부습도관측치',
       '펌프상태', '펌프작동남은시간', '최근분무량', '일간누적분무량', '냉방상태', '냉방작동남은시간', '난방상태',
       '난방작동남은시간', '내부유동팬상태', '내부유동팬작동남은시간', '외부환기팬상태', '외부환기팬작동남은시간',
       '화이트 LED상태', '화이트 LED작동남은시간', '화이트 LED동작강도', '레드 LED상태', '레드 LED작동남은시간',
       '레드 LED동작강도', '블루 LED상태', '블루 LED작동남은시간', '블루 LED동작강도', '카메라상태', '냉방온도',
       '난방온도', '기준온도', '난방부하', '냉방부하', '총추정광량', '백색광추정광량', '적색광추정광량',
       '청색광추정광량']

target_columns = [
    '내부온도관측치',
    '내부습도관측치',
    'CO2관측치',    
    'EC관측치',
    '외부온도관측치',
    '외부습도관측치',
    '최근분무량',
    '일간누적분무량',
    '화이트 LED동작강도',
    '레드 LED동작강도',
    '블루 LED동작강도',
    '냉방온도',
    '난방온도',
    '기준온도',
    '난방부하',
    '냉방부하', 
    '백색광추정광량',
    '적색광추정광량',
    '청색광추정광량',
    '월'
]

# train-test(valid) split
all_input_list  = sorted(glob.glob("/home/gyuseonglee/DS_practice_AB/train_input/*.csv"))
all_target_list = sorted(glob.glob("/home/gyuseonglee/DS_practice_AB/train_target/*.csv"))
all_data = [ (all_input_list[i], all_target_list[i]) for i in range(len(all_input_list)) ]
processed_X = []
processed_Y = []

# run preprocessing function
preprocessing(all_input_list, all_target_list, processed_X, processed_Y)
processed_X = torch.vstack(processed_X)
processed_Y = torch.vstack(processed_Y)

print("shapes of processed X and Y  : ")
print(processed_X.shape)
print(processed_Y.shape)
X_train_all, X_val_all, Y_train_all, Y_val_all = train_test_split(processed_X, processed_Y, test_size = 0.1, shuffle = False)


'''
        5. Set model and parameters
'''
class origin(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(origin, self).__init__()
        self.attention_forward = TransformerEncoder(
            num_layers = 4,
            num_heads  = 6,
            dim_model  = input_dim,
            dim_feedforward = hidden_dim,
            dropout = 0.1
        )
        self.attention_backward = TransformerEncoder(
            num_layers = 4,
            num_heads  = 6,
            dim_model  = input_dim,
            dim_feedforward = hidden_dim,
            dropout = 0.12
        )
        self.linear = nn.Linear(
            in_features  = input_dim,
            out_features = output_dim
        )
        

    def forward(self, x):        
        forward_x  = self.attention_forward(x)
        backward_x = self.attention_backward(x.flip(dims=[1]))
        con = forward_x + backward_x    
        x = self.linear(con)[:, -1, :]
        return x
        
    
# hyper parameters and data shapes
lr = 0.001
n_epochs = 100
max_norm = 10
N = 16   # batch_size
T = processed_X.shape[1]
I = processed_X.shape[2]
O = processed_Y.shape[1]  
H = 2048

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"current device : {device}")
print(f"lr             : {lr}")
print(f"n_epochs       : {n_epochs}")
print(f"max_norm       : {max_norm}\n")
print(f"N (batch_size)  : {N}")
print(f"T (time_step)   : {T}")
print(f"I (input_dim)   : {I}")
print(f"H (hidden_dim)  : {H}")
print(f"O (output_dim)  : {O}")

# empty cuda cache
torch.cuda.empty_cache()


# set model
model = origin(I, H, O)
criterion = nn.L1Loss()
optimizer = optim.Adam(
    params=model.parameters(), 
    lr = lr,
    eps = 1e-08,
    weight_decay = 0.1
)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
clip      = True

'''
        6. Run and save model
'''
best_model = train(
    model, 
    device, 
    criterion, 
    optimizer, 
    scheduler, 
    clip, 
    X_train_all, 
    Y_train_all, 
    X_val_all, 
    Y_val_all, 
    lr, 
    n_epochs, 
    N,
    max_norm
)
torch.save(best_model.state_dict(), "model_0907_AB_state")
torch.save(best_model, "model_0907_AB")


'''
        7. save results
'''
curmodel = torch.load("last_best_model")
# curmodel = best_model
input_list  = sorted(glob.glob("/home/gyuseonglee/DS_practice_AB/test_input/*.csv"))
target_list = sorted(glob.glob("/home/gyuseonglee/DS_practice_AB/test_target/*.csv"))

inputs = []
targets = []
for idx in range(6):
    temp = []
    x = pd.read_csv(input_list[idx]).drop_duplicates().reset_index()
    del x["index"]
    x.columns = origin_columns
    x["월"] = x["시간"].str.split("-", expand = True)[1].astype(int)
    x["월"] = np.where(
        x["월"]==12, 0, x["월"] 
    )
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


for i in range(6):
    with torch.no_grad():
        output = curmodel.forward(inputs[i].to("cuda")).to("cpu").squeeze(1)
        cur_target = pd.read_csv(target_list[i])
        cur_target["rate"] = output
        cur_target.to_csv(target_list[i], index = False)
