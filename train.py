import torch
import numpy as np
import argparse

from dataloader import PoiLoader, Split, Usage
from torch.utils.data import DataLoader
from setting import Setting
from trainer import FlashbackPlusPlusTrainer
from network import create_h0_strategy

### parse settings ###
setting = Setting()
setting.parse()
print(setting)


### load dataset ###
poi_loader = PoiLoader(setting.max_users, setting.min_checkins,setting.sequence_length)
poi_loader.load(setting.dataset_file)
dataset = poi_loader.poi_dataset(setting.sequence_length, setting.batch_size, Split.TRAIN, Usage.MAX_SEQ_LENGTH)
dataset_test = poi_loader.poi_dataset(setting.sequence_length, setting.batch_size, Split.TEST, Usage.MAX_SEQ_LENGTH)
dataloader = DataLoader(dataset, batch_size = 1, shuffle=False,drop_last=True)
dataloader_test = DataLoader(dataset_test, batch_size = 1, shuffle=False,drop_last=True)

# setup trainer
trainer = FlashbackPlusPlusTrainer(setting.lambda_t, setting.lambda_s)
trainer.prepare(poi_loader.locations(), poi_loader.user_count(), setting.hidden_dim, setting.rnn_factory, setting.device)
h0_strategy = create_h0_strategy(setting.hidden_dim, setting.is_lstm)
print('{} {}'.format(trainer.greeter(), setting.rnn_factory.greeter()))

optimizer = torch.optim.Adam(trainer.parameters(), lr = setting.learning_rate, weight_decay = setting.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.2)

def evaluate_test():
    dataset_test.reset()
    h = h0_strategy.on_init(setting.batch_size, setting.device)
    
    with torch.no_grad():        
        iter_cnt = 0
        recall1 = 0
        recall5 = 0
        recall10 = 0
        average_precision = 0.
        
        u_iter_cnt = np.zeros(poi_loader.user_count())
        u_recall1 = np.zeros(poi_loader.user_count())
        u_recall5 = np.zeros(poi_loader.user_count())
        u_recall10 = np.zeros(poi_loader.user_count())
        u_average_precision = np.zeros(poi_loader.user_count())        
        reset_count = torch.zeros(poi_loader.user_count())
        
        for i, (x, t, s, y, y_t, y_s, reset_h, active_users) in enumerate(dataloader_test):
            active_users = active_users.squeeze()
            for j, reset in enumerate(reset_h):
                if reset:
                    if setting.is_lstm:
                        hc = h0_strategy.on_reset_test(active_users[j], setting.device)
                        h[0][0, j] = hc[0]
                        h[1][0, j] = hc[1]
                    else:
                        h[0, j] = h0_strategy.on_reset_test(active_users[j], setting.device)
                    reset_count[active_users[j]] += 1

            
            # squeeze for reasons of "loader-batch-size-is-1"
            x = x.squeeze().to(setting.device)
            t = t.squeeze().to(setting.device)
            s = s.squeeze().to(setting.device)            
            y = y.squeeze()
            y_t = y_t.squeeze().to(setting.device)
            y_s = y_s.squeeze().to(setting.device)
            
            active_users = active_users.to(setting.device)            
        
            # evaluate:
            out, h = trainer.evaluate(x, t, s, y_t, y_s, h, active_users)
            
            for j in range(setting.batch_size):  
                # o contains a per user list of votes for all locations for each sequence entry
                o = out[j]
                
                    # partition elements
                o_n = o.cpu().detach().numpy()
                ind = np.argpartition(o_n, -10, axis=1)[:, -10:] # top 10 elements
                                    
                y_j = y[:, j]
                
                for k in range(len(y_j)):                    
                    if (reset_count[active_users[j]] > 1):
                        continue
                    
                    # resort indices for k:
                    ind_k = ind[k]
                    r = ind_k[np.argsort(-o_n[k, ind_k], axis=0)] # sort top 10 elements descending
                    
                    r = torch.tensor(r)
                    t = y_j[k]
                    
                    # compute MRR:
                    r_kj = o_n[k, :]
                    t_val = r_kj[t]
                    upper = np.where(r_kj > t_val)[0]
                    precision = 1. / (1+len(upper))
                    
                    # store
                    u_iter_cnt[active_users[j]] += 1
                    u_recall1[active_users[j]] += t in r[:1]
                    u_recall5[active_users[j]] += t in r[:5]
                    u_recall10[active_users[j]] += t in r[:10]
                    u_average_precision[active_users[j]] += precision
    
        formatter = "{0:.8f}"
        for j in range(poi_loader.user_count()):
            iter_cnt += u_iter_cnt[j]
            recall1 += u_recall1[j]
            recall5 += u_recall5[j]
            recall10 += u_recall10[j]
            average_precision += u_average_precision[j]

            if (setting.report_user > 0 and (j+1) % setting.report_user == 0):
                print('Report user', j, 'preds:', u_iter_cnt[j], 'recall@1', formatter.format(u_recall1[j]/u_iter_cnt[j]), 'MAP', formatter.format(u_average_precision[j]/u_iter_cnt[j]), sep='\t')
            
        print('recall@1:', formatter.format(recall1/iter_cnt))
        print('recall@5:', formatter.format(recall5/iter_cnt))
        print('recall@10:', formatter.format(recall10/iter_cnt))
        print('MAP', formatter.format(average_precision/iter_cnt))
        print('predictions:', iter_cnt)


# train!
for e in range(setting.epochs):
    h = h0_strategy.on_init(setting.batch_size, setting.device)
    
    dataset.shuffle_users() # shuffle users before each epoch!
    for i, (x, t, s, y, y_t, y_s, reset_h, active_users) in enumerate(dataloader):
        for j, reset in enumerate(reset_h):
            if reset:
                if setting.is_lstm:
                    hc = h0_strategy.on_reset(active_users[0][j])
                    h[0][0, j] = hc[0]
                    h[1][0, j] = hc[1]
                else:
                    h[0, j] = h0_strategy.on_reset(active_users[0][j])
        
        x = x.squeeze().to(setting.device)
        t = t.squeeze().to(setting.device)
        s = s.squeeze().to(setting.device)
        y = y.squeeze().to(setting.device)
        y_t = y_t.squeeze().to(setting.device)
        y_s = y_s.squeeze().to(setting.device)                
        active_users = active_users.to(setting.device)       
        optimizer.zero_grad()
        loss, h1 = trainer.loss(x, t, s, y, y_t, y_s, h, active_users)
        loss.backward(retain_graph=True) # backpropagate through time to adjust the weights and find the gradients of the loss function
        latest_loss = loss.item()        
        optimizer.step()
    scheduler.step()
    
    # statistics:
    if (e+1) % 1 == 0:
        print(f'Epoch: {e+1}/{setting.epochs}')
        print(f'Loss: {latest_loss}')
        print("alpha:",trainer.t_module.lambda_t.data.item(),",beta:",trainer.s_module.lambda_s.data.item())
    if (e+1) % setting.validate_epoch == 0:
        print('~~~ Test Set Evaluation ~~~')
        evaluate_test()




