import torch
from torch import nn
import numpy as np

from network import S_Module, T_Module, FlashbackPlusPlus



class FlashbackPlusPlusTrainer():
    
    def __init__(self, lambda_t, lambda_s):
        
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s
    
    def greeter(self):
        return 'Use Flashback++ training.'
    
    def debug(self):
        pass
    
    def parameters(self):
        params = list(self.model.parameters())\
            + list(self.t_module.parameters())\
            + list(self.s_module.parameters())
        return params
        
    
    def prepare(self, loc_count, user_count, hidden_size, RNNFactory, device):
        self.s_module = S_Module(self.lambda_s).to(device)
        self.t_module = T_Module(self.lambda_t).to(device)
        f_t = lambda delta_t: self.t_module(delta_t)
        f_s = lambda delta_s: self.s_module(delta_s)
            
        
        self.loc_count = loc_count
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.model = FlashbackPlusPlus(loc_count, user_count, hidden_size, f_t, f_s, RNNFactory).to(device)

    
    def evaluate(self, x, t, s, y_t, y_s, h, active_users):
        ''' takes a batch (users x location sequence)
        then does the prediction and returns a list of user x sequence x location
        describing the probabilities for each location at each position in the sequence.
        t, s are temporal and spatial data related to the location sequence x
        y_t, y_s are temporal and spatial data related to the target sequence y.
        Flashback does not access y_t and y_s for prediction!
        '''
        self.model.eval()
        out, h = self.model(x, t, s, y_t, y_s, h, active_users)
        out_t = out.transpose(0, 1)
        return out_t, h # model output is directly associated with the ranking per location.
    
    def loss(self, x, t, s, y, y_t, y_s, h, active_users):
        ''' takes a batch (users x location sequence)
        and corresponding targets in order to compute the training loss '''
        
        self.model.train()
        out, h = self.model(x, t, s, y_t, y_s, h, active_users)
        out = out.view(-1, self.loc_count)
        y = y.view(-1)
        l = self.cross_entropy_loss(out, y)
        return l, h
    
    
