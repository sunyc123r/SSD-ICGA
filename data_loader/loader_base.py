from cgi import test
import os
from ssl import PROTOCOL_TLS_CLIENT
from time import time
import random
import collections
import scipy.sparse as sp
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data

class DataLoaderBase(data.Dataset):

    def __init__(self, args, logging):
        self.args = args
        self.data_name = args.data_name
        self.device=args.device
        self.data_dir = os.path.join(args.data_dir, args.data_name)
        self.num_ng=args.neg_num

        self.train_file = os.path.join(self.data_dir, 'train.txt')
        self.valid_file = os.path.join(self.data_dir, 'valid.txt')
        self.test_file = os.path.join(self.data_dir, 'test.txt')

        self.social_file = os.path.join(self.data_dir, 'social.txt')
        
        self.cf_train_data, self.train_user_dict = self.load_data(self.train_file)
        self.cf_valid_data, self.valid_user_dict = self.load_data(self.valid_file)
        self.cf_test_data, self.test_user_dict = self.load_data(self.test_file)

        self.social_data, self.social_dict = self.load_data(self.social_file)
        self.statistic_cf()

    def ng_sample(self): 
        self.train_fill=[]
        for x in range(len(self.cf_train_data[0])):
            u, i = self.cf_train_data[0][x], self.cf_train_data[1][x]
            for t in range(self.num_ng):
                j = np.random.randint(self.n_items)
                while  j in self.train_user_dict[u]:
                    j = np.random.randint(self.n_items)
                self.train_fill.append([u, i, j])

    def __len__(self):     
        return self.num_ng * len(self.cf_train_data[0]) 
    
    def __getitem__(self,idx):
        user = self.train_fill[idx][0]
        item_i = self.train_fill[idx][1]
        item_j = self.train_fill[idx][2]
        return user, item_i, item_j 
    
    def load_data(self, filename):
        user = []
        item = []
        user_dict = dict()

        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:
                user_id, item_ids = inter[0], inter[1:]
                item_ids = list(set(item_ids))
                for item_id in item_ids:
                    user.append(user_id)
                    item.append(item_id)
                user_dict[user_id] = item_ids

        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)
        return (user, item), user_dict

    def statistic_cf(self):
        a=[max(self.cf_train_data[0]), max(self.cf_test_data[0]),max(self.cf_valid_data[0]),
        max(self.social_data[0]),max(self.social_data[1])]
        b=[max(self.cf_train_data[1]), max(self.cf_test_data[1]),max(self.cf_valid_data[1])]
        self.n_users = max(a) + 1
        self.n_items = max(b) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_valid = len(self.cf_valid_data[0])
        self.n_cf_test = len(self.cf_test_data[0])
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users.tolist():
            #print(user)
            posItems.append(self.train_user_dict[user])
        return posItems

    
    

    

