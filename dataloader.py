import torch
from torch.utils.data import Dataset
from enum import Enum
import random
from datetime import datetime

class Split(Enum):
    TRAIN = 0
    TEST = 2
    USE_ALL = 3

class Usage(Enum):
    MIN_SEQ_LENGTH = 0
    MAX_SEQ_LENGTH = 1
    CUSTOM = 2
    

class PoiDataset(Dataset):
    '''
    Our Point-of-interest pytorch dataset: To maximize GPU workload we organize the data in batches of
    "user" x "a fixed length sequence of locations". The active users have at least one sequence in the batch.
    In order to fill the batch all the time we wrap around the available users: if an active user
    runs out of locations we replace him with a new one. When there are no unused users available
    we reuse already processed ones. This happens if a single user was way more active than the average user.
    The batch guarantees that each sequence of each user was processed at least once.
    
    This data management has the implication that some sequences might be processed twice (or more) per epoch.
    During trainig you should call PoiDataset::shuffle_users before the start of a new epoch. This
    leads to more stochastic as different sequences will be processed twice.
    During testing you *have to* keep track of the already processed users.    
    
    Working with a fixed sequence length omits awkward code by removing only few of the latest checkins per user.
    We work with a 80/20 train/test spilt, where test check-ins are strictly after training checkins.
    To obtain at least one test sequence with label we require any user to have at least (5*<sequence-length>+1) checkins in total.    
    '''    
    def reset(self):
        # reset training state:
        self.next_user_idx = 0 # current user index to add
        self.active_users = [] # current active users
        self.active_user_seq = [] # current active users sequences
        self.user_permutation = [] # shuffle users during training
        self.removed_users=[] #removed users that all of the sequences longger than 120 days

        # set active users:
        for i in range(self.user_length):
            self.next_user_idx += 1
            self.active_users.append(i) 
            self.active_user_seq.append(0)
        
        # use 1:1 permutation:
        for i in range(len(self.users)):
            self.user_permutation.append(i)

        
    def shuffle_users(self):
        random.shuffle(self.user_permutation)    
        # reset active users:
        self.next_user_idx = 0
        self.active_users = []
        self.active_user_seq = []
        for i in range(self.user_length):
            self.next_user_idx += 1
            self.active_users.append(self.user_permutation[i]) 
            self.active_user_seq.append(0)
    
    def __init__(self, users, times, coords, locs, seq_length, user_length, split, usage, loc_count, custom_seq_count):
        self.users = users
        self.times = times
        self.coords = coords
        self.locs = locs
        self.labels = []
        self.lbl_times = []
        self.lbl_coords = []
        self.sequences = []
        self.sequences_times = []
        self.sequences_coords = []
        self.sequences_labels = []
        self.sequences_lbl_times = []
        self.sequences_lbl_coords = []
        self.sequences_count = []
        self.Ps = []
        self.Qs = torch.zeros(loc_count, 1)
        self.usage = usage
        self.user_length = user_length
        self.loc_count = loc_count
        self.custom_seq_count = custom_seq_count
        #self.loc_ids_of_user = [] # sets of unique locations per user

        self.reset()

        # collect locations:
        for i in range(loc_count):
            self.Qs[i, 0] = i    
        
        # align labels to locations
        for i, loc in enumerate(locs):
            self.locs[i] = loc[:-1]
            self.labels.append(loc[1:])
            # adapt time and coords:
            self.lbl_times.append(self.times[i][1:])
            self.lbl_coords.append(self.coords[i][1:])
            self.times[i] = self.times[i][:-1]
            self.coords[i] = self.coords[i][:-1]
        
        # split to training / test phase:
        for i, (time, coord, loc, label, lbl_time, lbl_coord) in enumerate(zip(self.times, self.coords, self.locs, self.labels, self.lbl_times, self.lbl_coords)):
            train_thr = int(len(loc) * 0.8)
            if (split == Split.TRAIN):
                self.times[i] = time[:train_thr]
                self.coords[i] = coord[:train_thr]
                self.locs[i] = loc[:train_thr]
                self.labels[i] = label[:train_thr]
                self.lbl_times[i] = lbl_time[:train_thr]
                self.lbl_coords[i] = lbl_coord[:train_thr]
            if (split == Split.TEST):
                self.times[i] = time[train_thr:]
                self.coords[i] = coord[train_thr:]
                self.locs[i] = loc[train_thr:]
                self.labels[i] = label[train_thr:]
                self.lbl_times[i] = lbl_time[train_thr:]
                self.lbl_coords[i] = lbl_coord[train_thr:]
            if (split == Split.USE_ALL):
                pass # do nothing
            
        # split location and labels to sequences:
        self.max_seq_count = 0
        self.min_seq_count = 10000000
        self.capacity = 0
        RemoveFlag=True #do not train the squence longger than 120 days 
        for i, (time, coord, loc, label, lbl_time, lbl_coord) in enumerate(zip(self.times, self.coords, self.locs, self.labels, self.lbl_times, self.lbl_coords)):
            seq_count = len(loc) // seq_length
            assert seq_count > 0 # fix seq_length and min-checkins in order to have test sequences in a 80/20 split!
            seqs = []
            seq_times = []
            seq_coords = []
            seq_lbls = []
            seq_lbl_times = []
            seq_lbl_coords = []
            fixed=0 #remove corner case
            for j in range(seq_count):
                start = j * seq_length
                end = (j+1) * seq_length
                if(RemoveFlag and time[end-1]-time[start]>10368000 and split == Split.TRAIN ):
                    fixed+=1
                    continue
                seqs.append(loc[start:end])
                seq_times.append(time[start:end])
                seq_coords.append(coord[start:end])
                seq_lbls.append(label[start:end])
                seq_lbl_times.append(lbl_time[start:end])
                seq_lbl_coords.append(lbl_coord[start:end])
            seq_count-=fixed
            self.sequences.append(seqs)
            self.sequences_times.append(seq_times)
            self.sequences_coords.append(seq_coords)            
            self.sequences_labels.append(seq_lbls)
            self.sequences_lbl_times.append(seq_lbl_times)
            self.sequences_lbl_coords.append(seq_lbl_coords)
            self.sequences_count.append(seq_count)
            self.capacity += seq_count
            self.max_seq_count = max(self.max_seq_count, seq_count)
            self.min_seq_count = min(self.min_seq_count, seq_count)
        
        # statistics
        if (self.usage == Usage.MIN_SEQ_LENGTH):
            print(split,'load', len(users), 'users with min_seq_count', self.min_seq_count, 'batches:', self.__len__())
        if (self.usage == Usage.MAX_SEQ_LENGTH):
            print(split,'load', len(users), 'users with max_seq_count', self.max_seq_count, 'batches:', self.__len__())
        if (self.usage == Usage.CUSTOM):
            print(split,'load', len(users), 'users with custom_seq_count', self.custom_seq_count, 'Batches:', self.__len__())
            
    
    def sequences_by_user(self, idx):
        return self.sequences[idx]
    
    def __len__(self):
        if (self.usage == Usage.MIN_SEQ_LENGTH):
            # min times amount_of_user_batches:
            return self.min_seq_count * (len(self.users) // self.user_length)
        if (self.usage == Usage.MAX_SEQ_LENGTH):
            # estimated capacity:
            estimated = self.capacity // self.user_length
            return max(self.max_seq_count, estimated)
        if (self.usage == Usage.CUSTOM):
            return self.custom_seq_count * (len(self.users) // self.user_length)
        raise Exception('Piiiep')
    
    def __getitem__(self, idx):
        ''' Against pytorch convention, we directly build a full batch inside __getitem__.
        Use a batch_size of 1 in your pytorch data loader.
        
        A batch consists of a list of active users,
        their next location sequence with timestamps and coordinates.
        
        y is the target location and y_t, y_s the targets timestamp and coordiantes. Provided for
        possible use.
        
        reset_h is a flag which indicates when a new user has been replacing a previous user in the
        batch. You should reset this users hidden state to initial value h_0.
        '''
        seqs = []
        times = []
        coords = []
        lbls = []
        lbl_times = []
        lbl_coords = []
        reset_h = []
        for i in range(self.user_length):
            i_user = self.active_users[i]
            j = self.active_user_seq[i]
            max_j = self.sequences_count[i_user]
            if (self.usage == Usage.MIN_SEQ_LENGTH):
                max_j = self.min_seq_count
            if (self.usage == Usage.CUSTOM):
                max_j = min(max_j, self.custom_seq_count) # use either the users maxima count or limit by custom count
            if (j >= max_j ):
                # repalce this user in current sequence:
                i_user = self.user_permutation[self.next_user_idx]
                j = 0
                self.active_users[i] = i_user
                self.active_user_seq[i] = j
                self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                while self.user_permutation[self.next_user_idx] in self.active_users:
                    self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                # TODO: throw exception if wrapped around!
            # use this user:
            reset_h.append(j == 0)
            seqs.append(torch.tensor(self.sequences[i_user][j]))
            times.append(torch.tensor(self.sequences_times[i_user][j]))
            coords.append(torch.tensor(self.sequences_coords[i_user][j]))
            lbls.append(torch.tensor(self.sequences_labels[i_user][j]))
            lbl_times.append(torch.tensor(self.sequences_lbl_times[i_user][j]))
            lbl_coords.append(torch.tensor(self.sequences_lbl_coords[i_user][j]))
            self.active_user_seq[i] += 1

        x = torch.stack(seqs, dim=1)
        t = torch.stack(times, dim=1)
        s = torch.stack(coords, dim=1)
        y = torch.stack(lbls, dim=1)
        y_t = torch.stack(lbl_times, dim=1)
        y_s = torch.stack(lbl_coords, dim=1)           
        return x, t, s, y, y_t, y_s, reset_h, torch.tensor(self.active_users) #, P, poi2id


class PoiLoader():
    
    def __init__(self, max_users = 0, min_checkins = 0, seq_length=0):
        self.max_users = max_users
        self.min_checkins = min_checkins
        self.seq_length=seq_length
        self.user2id = {}
        self.poi2id = {}
        
        self.users = []
        self.times = []
        self.coords = []
        self.locs = []
    
    def poi_dataset(self, seq_length, user_length, split, usage, custom_seq_count = 1):
        dataset = PoiDataset(self.users.copy(), self.times.copy(), self.coords.copy(), self.locs.copy(), seq_length, user_length, split, usage, len(self.poi2id), custom_seq_count) # crop latest in time
        return dataset
    
    def locations(self):
        return len(self.poi2id)

    def user_count(self):
        return len(self.users)   
        
    def load(self, file):
        # collect all users with min checkins:
        self.load_users(file)
        # collect checkins for all collected users:
        self.load_pois(file)
    
    def load_users(self, file):
        f = open(file, 'r')
        lines = f.readlines()
    
        prev_user = int(lines[0].split('\t')[0])
        visit_cnt = 0
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if user == prev_user:
                visit_cnt += 1
            else:
                if visit_cnt >= self.min_checkins:
                    self.user2id[prev_user] = len(self.user2id)
                prev_user = user
                visit_cnt = 1
                if self.max_users > 0 and len(self.user2id) >= self.max_users:
                    break # restrict to max users
    
    def load_pois(self, file):
        f = open(file, 'r')
        lines = f.readlines()
        
        # store location ids
        user_time = []
        user_coord = []
        user_loc = []
        
        prev_user = int(lines[0].split('\t')[0])
        prev_user = self.user2id.get(prev_user)
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if self.user2id.get(user) is None:
                continue # user is not of interrest
            user = self.user2id.get(user)
            time = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ") - datetime(1970, 1, 1)).total_seconds() # unix seconds
            lat = float(tokens[2]) # WGS84? Latitude
            long = float(tokens[3]) # WGS84? Longitude
            coord = (lat, long)

            location = int(tokens[4]) # location nr
            if self.poi2id.get(location) is None: # get-or-set locations
                self.poi2id[location] = len(self.poi2id)
            location = self.poi2id.get(location)
    
            if user == prev_user:
                # insert in front!
                user_time.insert(0, time)
                user_coord.insert(0, coord)
                user_loc.insert(0, location)
            else:
                self.users.append(prev_user)
                self.times.append(user_time)
                self.coords.append(user_coord)
                self.locs.append(user_loc)
                
                # resart:
                prev_user = user 
                user_time = [time]
                user_coord = [coord]
                user_loc = [location] 
                
        # process also the latest user in the for loop
        self.users.append(prev_user)
        self.times.append(user_time)
        self.coords.append(user_coord)
        self.locs.append(user_loc)