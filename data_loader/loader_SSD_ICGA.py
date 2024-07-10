
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from time import time
import math
import json
import os
import random
from scipy.sparse import coo_matrix, csr_matrix

from data_loader.loader_base import DataLoaderBase

from scipy.sparse import find
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import networkx as nx


def page_rank(n_users,head,tail):
    ## page rank
    G = nx.DiGraph()
    # Add nodes from both lists
    all_nodes =list(range(n_users))
    G.add_nodes_from(all_nodes)
    # Add directed edges based on the lists
    for start, end in zip(head,tail):
        G.add_edge(start, end)
    pr = nx.pagerank(G, alpha=0.9)
    return pr


class DataLoader_SSD_ICGA(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.deivce=self.args.device
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size

        self.inter_graph,self.inter_norm=self.getRatingAdjacency()
        self.social_graph,self.social_norm=self.getSocialAdjacency()

        self.social_inverse=self.social_dict_inverse()  # trust by who
        self.social_user=np.array(list(self.social_inverse.keys()))
        self.n_social=len(self.social_inverse.keys())

        self.pr=self.get_pr()
        self.importance_score=np.array(self.pr)[self.social_user]
        self.importance_score= self.importance_score / np.sum(self.importance_score)

        self.kol_idx=[]
        idx=0
        sort_pr = sorted(range(len(self.pr)), key=lambda i: self.pr[i], reverse=True)

        while len(self.kol_idx)<int(self.n_social*args.activation_ratio):
            if sort_pr[idx] in self.social_user:
                self.kol_idx.append(sort_pr[idx])
            idx+=1

        self.social_wokol=self.get_wokol_social()
        self.cl_sample()
        self.print_info(logging)

    def cl_sample(self):

        self.cl_dict={}
        for i in self.social_dict.keys():
                
                trust_neighbor=self.social_dict[i]
                ub_kol=list(set(self.kol_idx).intersection(set(trust_neighbor)))
                if len(ub_kol)>0:
                    ua=[i]*len(ub_kol)
                    idx=random.sample(list(range(len(self.social_wokol[0]))),len(ub_kol))
                    normal_a=self.social_wokol[0][idx]
                    normal_b=self.social_wokol[1][idx]

                    ub_unobserved=[]
                    for j in range(len(normal_a)):

                        b=random.sample(list(self.social_inverse.keys()),1)
                        cond= b not in self.kol_idx and b not in self.social_dict[normal_a[j]] #and b in self.data.social_dict.keys()
                        while not cond:
                            b=random.sample(list(self.social_inverse.keys()),1)
                        ub_unobserved.extend(b)
        
                    self.cl_dict[i]=[ua,ub_kol,normal_a,normal_b,ub_unobserved]
    
    def get_wokol_social(self):
        head=[]
        tail=[]
        for i in range(len(self.social_data[0])):
            if self.social_data[1][i] not in self.kol_idx:
                head.append(self.social_data[0][i])
                tail.append(self.social_data[1][i])
        return (np.array(head, dtype=np.int32),np.array(tail, dtype=np.int32))


    def get_pr(self):

        name="/pr.pickle"
        if os.path.exists(self.data_dir+name):
            print("pr Loading")
            with open(self.data_dir+name, 'rb') as handle:
                pr = pickle.load(handle)
        else:
            print("calculating pr")
            pr=[]
            pr_dict=page_rank(self.n_users,self.social_data[0],self.social_data[1])
            for i in range(self.n_users):
                pr.append(pr_dict[i])
            with open(self.data_dir+name, 'wb') as handle:
                pickle.dump(pr, handle, protocol=pickle.HIGHEST_PROTOCOL)  
        return pr
        

    def social_dict_inverse(self):
        social_inverse={}
        for i in range(len(self.social_data[0])):
            ua=self.social_data[0][i]
            ub=self.social_data[1][i]
            if ub in social_inverse.keys():
                social_inverse[ub].append(ua)
            else:
                social_inverse[ub]=[ua]
        return social_inverse

    
    def getRatingAdjacency(self):
        try:
            t1=time()
            inter_graph = sp.load_npz(self.data_dir + '/inter_adj.npz')
            inter_norm = sp.load_npz(self.data_dir + '/inter_norm.npz')
            print('already load adj matrix', inter_graph.shape, time() - t1)

        except Exception:
            self.train_item_dict=self.create_item_dict()
            inter_graph,inter_norm = self.buildRatingAdjacency()
            sp.save_npz(self.data_dir + '/inter_adj.npz', inter_graph)
            sp.save_npz(self.data_dir + '/inter_norm.npz', inter_norm)
        
        return inter_graph,inter_norm

    def getSocialAdjacency(self):
        
        try:
            t1=time()
            social_graph = sp.load_npz(self.data_dir + '/social_adj.npz')
            social_norm = sp.load_npz(self.data_dir + '/social_norm.npz')
            print('already load social matrix', social_graph.shape, time() - t1)
           
        except Exception:
            print('creating')
            social_graph,social_norm = self.buildSocialAdjacency()
        
            sp.save_npz(self.data_dir + '/social_adj.npz', social_graph)
            sp.save_npz(self.data_dir + '/social_norm.npz', social_norm)
            
        return social_graph,social_norm 

    def buildRatingAdjacency(self):
        row, col, entries, norm_entries = [], [], [], []
        train_h_list,train_t_list = self.cf_train_data[0], self.cf_train_data[1]

        for i in range(len(train_h_list)):
            user=train_h_list[i]
            item=train_t_list[i]
            row += [user,item+self.n_users]
            col += [item+self.n_users,user]
            entries+=[1,1]
            degree=1 / math.sqrt(len(self.train_user_dict[user])) /math.sqrt(len(self.train_item_dict[item]))
            norm_entries += [degree,degree]
        entries=np.array(entries)
        user=np.array(row)
        item=np.array(col)

        adj = coo_matrix((entries, (user, item)),shape=(self.n_users+self.n_items,self.n_users+self.n_items))
        norm_adj = coo_matrix((norm_entries, (user, item)),shape=(self.n_users+self.n_items, self.n_users+self.n_items))

        return adj, norm_adj

    def buildSocialAdjacency(self):
        row, col, entries, norm_entries = [], [], [], []
        train_h_list,train_t_list = self.social_data[0], self.social_data[1]
        for i in range(len(train_h_list)):

            user=train_h_list[i]
            item=train_t_list[i]
            row += [user]
            col += [item]
            entries+=[1]
            if item in self.social_dict.keys():
                div=len(self.social_dict[item])
            else:
                div=1
            norm_entries += [1 / math.sqrt(len(self.social_dict[user])) /
            math.sqrt(div)]
        entries=np.array(entries)
        user=np.array(row)
        item=np.array(col)

        adj = coo_matrix((entries, (user, item)),shape=(self.n_users, self.n_users))
        norm_adj = coo_matrix((norm_entries, (user, item)),shape=(self.n_users, self.n_users))

        return adj, norm_adj
    
    def print_info(self, logging):
        logging.info('n_users:     %d' % self.n_users)
        logging.info('n_items:     %d' % self.n_items)
        logging.info('n_cf_train:  %d' % self.n_cf_train)
        logging.info('n_cf_test:   %d' % self.n_cf_test)

