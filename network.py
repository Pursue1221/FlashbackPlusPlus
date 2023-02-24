import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum


class RNN(Enum):
    GRU = 0
    RNN = 1
    LSTM = 2
    
    @staticmethod
    def from_string(name):
        if name == 'gru':
            return RNN.GRU
        if name == 'rnn':
            return RNN.RNN
        if name == 'lstm':
            return RNN.LSTM
        raise ValueError('{} not supported'.format(name))
        

class RNNFactory():
    
    def __init__(self, rnn_type_str):
        self.rnn_type = RNN.from_string(rnn_type_str)
    
    def is_lstm(self):
        return self.rnn_type in [RNN.LSTM]
    
        
    def greeter(self):
        if self.rnn_type == RNN.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == RNN.RNN:
            return 'Use vanilla RNN implementation.'
        if self.rnn_type == RNN.LSTM:
            return 'Use pytorch LSTM implementation.'
        
    def create(self, hidden_size):
        if self.rnn_type == RNN.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == RNN.RNN:
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == RNN.LSTM:
            return nn.LSTM(hidden_size, hidden_size)
        
class S_Module(nn.Module):

    def __init__(self,beta):
        super(S_Module, self).__init__()
        if(beta==0):
            self.lambda_s = nn.Parameter(torch.rand(1))
        else:
            self.lambda_s = nn.Parameter(torch.ones(1)*beta)

    def forward(self, delta_s):
        return torch.exp(-(delta_s*self.lambda_s))



class T_Module(nn.Module):

    def __init__(self,alpha):
        super(T_Module, self).__init__()
        if(alpha==0):
            self.lambda_t = nn.Parameter(torch.rand(1))
        else:
            self.lambda_t = nn.Parameter(torch.ones(1)*alpha) 

    def forward(self, delta_t):
        return ((torch.cos(delta_t*2*np.pi/86400) + 1) / 2)*torch.exp(-(delta_t/86400*self.lambda_t))





class FlashbackPlusPlus(nn.Module):
    ''' GRU based rnn. applies weighted average using spatial and temporal data WITH user embeddings'''
    
    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, RNNFactory):
        super().__init__()
        self.input_size = input_size
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.f_t = f_t # function for computing temporal weight
        self.f_s = f_s # function for computing spatial weight

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.user_encoder = nn.Embedding(user_count, hidden_size)
        self.rnn = RNNFactory.create(hidden_size)
        self.fc = nn.Linear(2*hidden_size, input_size) # create outputs in lenght of locations

    def forward(self, x, t, s, y_t, y_s, h, active_user):        
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)        
        out, h = self.rnn(x_emb, h)
        
        # comopute weights per user
        out_w = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)
        for i in range(seq_len):
            sum_w = torch.zeros(user_len, 1, device=x.device)
            for j in range(i+1):
                dist_t = t[i] - t[j]
                dist_s = torch.norm(s[i] - s[j], dim=-1)
                a_j = self.f_t(dist_t)
                b_j = self.f_s(dist_s)
                a_j = a_j.unsqueeze(1)
                b_j = b_j.unsqueeze(1)
                w_j = a_j*b_j + 1e-10 # small epsilon to have no 0 division
                sum_w += w_j
                out_w[i] += w_j*out[j] # could be factored out into a matrix!
            # normliaze according to weights
            out_w[i] /= sum_w
        
        # add user embedding:
        p_u = self.user_encoder(active_user)
        p_u = p_u.view(user_len, self.hidden_size)
        out_pu = torch.zeros(seq_len, user_len, 2*self.hidden_size, device=x.device)
        for i in range(seq_len):
            out_pu[i] = torch.cat([out_w[i], p_u], dim=1)
        y_linear = self.fc(out_pu)
        return y_linear, h
        
'''
~~~ h_0 strategies ~~~
Initialize RNNs hidden states
'''

def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:        
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))        
    else:        
        return FixNoiseStrategy(hidden_size)

class H0Strategy():
    
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
    
    def on_init(self, user_len, device):
        pass
    
    def on_reset(self, user):
        pass
    
    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    ''' use fixed normal noise as initialization '''
    
    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1/self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu
    
    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)
    
    def on_reset(self, user):
        return self.h0

class LstmStrategy(H0Strategy):
    ''' creates h0 and c0 using the inner strategy '''
    
    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy
    
    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return (h,c)
    
    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return (h,c)
