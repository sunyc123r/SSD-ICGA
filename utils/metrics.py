import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
import random
from tqdm import tqdm
import pandas as pd

def metric_to_df(metric_dict,Ks):
    metric_names = ['ndcg','recall','precision']
    metric_col=['K']+metric_names
    ndcg=[];recall=[];precision=[]
    k=[]
    for i in Ks:
        k.append([i])
        recall.append(metric_dict[i]['recall'])
        ndcg.append(metric_dict[i]['ndcg'])
        precision.append(metric_dict[i]['precision'])
    metrics_df=pd.DataFrame([k,ndcg,recall,precision]).transpose()
    metrics_df.columns=metric_col
    return metrics_df

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def RecallPrecision_ATk(test_data, r, k):
    
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}

def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def test_one_batch(X,Ks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in Ks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}

def Valid(model,data,args,device):
    u_batch_size = data.test_batch_size
    testDict=data.valid_user_dict

    Ks = eval(args.Ks)
    max_K = max(Ks)
    metric_names = [ 'recall', 'ndcg','precision']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    # eval mode with no dropout
    model = model.eval()
    results = {'precision': np.zeros(len(Ks)),
               'recall': np.zeros(len(Ks)),
               'ndcg': np.zeros(len(Ks))}

    with torch.no_grad():
        user_ids = list(testDict.keys())
        user_ids_batches = [user_ids[i: i + u_batch_size] for i in range(0, len(user_ids), u_batch_size)]
        user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

        users_list = []
        rating_list = []
        groundTrue_list = []

        with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
            for batch_users in user_ids_batches:
                allPos = data.getUserPosItems(batch_users)
                groundTrue = [testDict[u] for u in batch_users.tolist()]
                batch_users_gpu=batch_users.to(device)
                #print(1)
                rating = model.getUsersRating(batch_users_gpu)
                #print(2)
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1<<10)
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
                pbar.update(1)

        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x,Ks))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(user_ids))
        results['precision'] /= float(len(user_ids))
        results['ndcg'] /= float(len(user_ids))

        for i in range(len(Ks)):
            for m in metric_names:
                metrics_dict[Ks[i]][m] = results[m][i]
        return metrics_dict
        
            
def Test(model,data,args,device):
    u_batch_size = data.test_batch_size
    testDict=data.test_user_dict

    Ks = eval(args.Ks)
    max_K = max(Ks)
    metric_names = [ 'recall', 'ndcg','precision']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    # eval mode with no dropout
    model = model.eval()
    results = {'precision': np.zeros(len(Ks)),
               'recall': np.zeros(len(Ks)),
               'ndcg': np.zeros(len(Ks))}
    
    with torch.no_grad():
        user_ids = list(testDict.keys())
        user_ids_batches = [user_ids[i: i + u_batch_size] for i in range(0, len(user_ids), u_batch_size)]
        user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

        users_list = []
        rating_list = []
        groundTrue_list = []

        with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
            for batch_users in user_ids_batches:
                allPos = data.getUserPosItems(batch_users)
                groundTrue = [testDict[u] for u in batch_users.tolist()]
                batch_users_gpu=batch_users.to(device)
                rating = model.getUsersRating(batch_users_gpu)
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1<<10)
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
                pbar.update(1)

        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x,Ks))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(user_ids))
        results['precision'] /= float(len(user_ids))
        results['ndcg'] /= float(len(user_ids))

        for i in range(len(Ks)):
            for m in metric_names:
                metrics_dict[Ks[i]][m] = results[m][i]
       
        return metrics_dict