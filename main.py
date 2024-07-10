import os
import sys

import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data as torch_data

from model.SSD_ICGA import SSD_ICGA
from parsers.parse_SSD_ICGA import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_SSD_ICGA import DataLoader_SSD_ICGA

def train(args):

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = args.device

    # load data
    data = DataLoader_SSD_ICGA(args, logging)
    train_loader = torch_data.DataLoader(data,batch_size=args.train_batch_size)

    model = SSD_ICGA(data,args)
    model.to(device)
    logging.info(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1
    best_hr = 0
    Ks = eval(args.Ks)
    k_min = 10
    k_max = 20

    epoch_list = []
    metrics_list = {k: {'ndcg': [],'precision':[],'recall':[]} for k in Ks}
    epoch=0
    should_stop=False
   
    while not should_stop:
        epoch+=1
        print('Start Epoch{}'.format(epoch))
        print('=================')
        time1 = time()
        model.train()
        train_loader.dataset.ng_sample()
        total_loss = 0
        iter=0
        n_batch = data.n_train // data.train_batch_size + 1
        time2 = time()

        for batch_user, batch_pos_item, batch_neg_item in train_loader:
            iter+=1
            batch_user = batch_user.long().to(device)
            batch_pos_item = batch_pos_item.long().to(device)
            batch_neg_item = batch_neg_item.long().to(device)

            bpr_batch_loss,reg_loss,ib_loss,ssl_loss,ent_loss= model.bpr_loss(batch_user, batch_pos_item, batch_neg_item)
            batch_loss = bpr_batch_loss+reg_loss*args.lambda1+ib_loss*args.lambda2+ssl_loss*args.lambda3+ent_loss*args.lambda4
            total_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (iter % args.print_every) == 0:
                print(ssl_loss.item())
                logging.info('Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Loss {:.4f} | total Mean Loss {:.4f}'
                .format(epoch, iter, n_batch, time() - time2,batch_loss.item(), total_loss / iter))
                time2 = time()

        logging.info('Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Mean Loss {:.4f}|'
        .format(epoch, n_batch, time() - time1, total_loss / n_batch))
    
        ## evaluation
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time3 = time()
            metrics_dict = Valid(model, data, args, device)
            logging.info('Evaluation: Epoch {:04d} | Total Time {:.1f}s |  NDCG [{:.4f}, {:.4f}], Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}]'.format(
                epoch, time() - time3, metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']
                , metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision']\
                , metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall']))\

            best_hr, should_stop = early_stopping(metrics_list[k_min]['ndcg'], args.stopping_steps)
            if should_stop:
                break
            if metrics_list[k_min]['ndcg'].index(best_hr) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, 'inter',best_epoch)
                logging.info('Save inter_model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch
        print('End of Epoch{}, total time {}s'.format(epoch,time()-time1))
        print('=======================')
    
    return best_epoch
  
 
def predict(args,best_epoch):
    # GPU / CPU
    device = args.device

    # load data
    data = DataLoader_SSD_ICGA(args, logging)

    # load model
    model = SSD_ICGA(data,args)
    model = load_model(model, args.save_dir+'inter_model_epoch{}.pth'.format(best_epoch))
    model.to(device)

    # predict
    Ks = eval(args.Ks)
    k_min = 10
    k_max = 20
   
    metrics_dict = Test(model, data, args, device)
    metric_df=metric_to_df(metrics_dict,Ks)
    metric_df.to_csv(args.save_dir + '/test_int_metrics.tsv', sep='\t', index=False)
    print('Test:  NDCG [{:.4f}, {:.4f}], precision[{:.4f}, {:.4f}],recall[{:.4f}, {:.4f}]'.format(
    metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']\
    , metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision']\
    , metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall']))
    
if __name__ == '__main__':
    args = parse_SSD_ICGA_args()
    best=train(args)
    predict(args,best)
                                    