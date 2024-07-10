import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import math
import random
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sp

##information bottleneck  

def temp_sigmoid(x,theta,yita=1):
    # theta is the temperature  #yita is the scale
    return yita / (1 + torch.exp(-x / theta))

def inner_product(a, b):
    return torch.sum(a*b, dim=-1)

def to_tensor(coo_mat,args):
    values = coo_mat.data
    indices = np.vstack((coo_mat.row, coo_mat.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo_mat.shape
    tensor_sparse=torch.sparse.FloatTensor(i, v, torch.Size(shape))
    tensor_sparse=tensor_sparse.to(args.device)
    return tensor_sparse

class MLP(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(MLP, self).__init__()
        self.linear1=nn.Linear(in_dim,int(in_dim/2))
        self.linear2=nn.Linear(int(in_dim/2),out_dim)
        self.relu=nn.ReLU()

    def forward(self,input):
        out=self.linear1(input)
        out=self.relu(out)
        out=self.linear2(out)
        return out

class LightGCN(nn.Module):

    def __init__(self,layer):
        super(LightGCN, self).__init__()
        self.n_layers=layer

    def forward(self,norm_adj,embed):
        all_embs = [embed]
        for layer in range(self.n_layers):
            if layer==0:
                all_embed = torch.sparse.mm(norm_adj, embed)
            else:
                all_embed = torch.sparse.mm(norm_adj, all_embed)

            all_embs.append(all_embed)
        embs = torch.stack(all_embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        return light_out , all_embs

class WeightGCN(nn.Module):

    def __init__(self,layer):
        super(WeightGCN, self).__init__()
        self.n_layers=layer

    def forward(self,graph,embed):
        all_embs = [embed]

        all_graph=[graph]
        for i in range(self.n_layers-1):
            all_graph.append(torch.sparse.mm(all_graph[i],graph))

        for layer in range(self.n_layers):
            all_embed = torch.sparse.mm(torch.sparse.softmax(all_graph[layer],dim=1),embed)  
            all_embs.append(all_embed)

        embs = torch.stack(all_embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        return light_out , all_embs
    
class SSD_ICGA(nn.Module):

    def __init__(self, data,config):

        super(SSD_ICGA, self).__init__()
        self.config = config
        self.data = data
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.inter_layers = config.inter_layers
        self.social_layers = config.social_layers
        self.embed_size = config.embed_dim

        self.act_ratio=config.activation_ratio  ##initial seed set ratio
        self.max_iterations=config.max_iterations
        self.sig_temp=config.sig_temp  ## sigmoid temperature
        self.ssl_temp = config.ssl_temp ##infonce loss hyper parameter
        self.yita=config.yita

        self.user_embedding = nn.Embedding(self.n_users,self.embed_size)
        self.item_embedding = nn.Embedding(self.n_items,self.embed_size)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        self.social_graph = to_tensor(data.social_graph,config)
        self.norm_inter_graph = to_tensor(data.inter_norm,config) 
        self.norm_social_graph = to_tensor(data.social_norm,config)

        #user modelling
        self.ui_encoder  = LightGCN(self.inter_layers)
        self.uu_encoder1 = LightGCN(self.social_layers) ##original graph (unweighted)
        self.uu_encoder2 = WeightGCN(self.social_layers) ## denoised graph (weighted)

        self.activation_func=MLP(2*self.embed_size,1)
        self.activater_transform=nn.Linear(self.embed_size,self.embed_size,bias=False)
        self.activatee_transform=nn.Linear(self.embed_size,self.embed_size,bias=False)

    def get_head_tail(self,graph,nodes):

        residual_graph=self.data.social_graph-graph ## edges that haven't been activated yet
        row_indices, col_indices = residual_graph.nonzero()

        head=[];tail=[]
        for  node in nodes:
            tail.extend(list(row_indices[col_indices == node]))  # be activated (trust)
            head.extend(list(col_indices[col_indices == node]))  # activate (be trusted)
        return torch.tensor(head,dtype=torch.int32).to(self.config.device), torch.tensor(tail,dtype=torch.int32).to(self.config.device)

    def get_weighted_graph(self,graph):
        row_indices, col_indices = graph.nonzero()
        _,weight_data=self.infer_activation(torch.tensor(col_indices).to(self.config.device),torch.tensor(row_indices).to(self.config.device))
        weight_graph =torch.sparse_coo_tensor((row_indices, col_indices), weight_data, [self.n_users, self.n_users])
        return weight_graph

    def infer_activation(self,h,t): 
        input=torch.cat((self.activater_transform(self.user_embedding(h)),self.activatee_transform(self.user_embedding(t))),1)
        p=self.activation_func(input) 
        p=p.squeeze(1)
        normalized_tensor=torch.sigmoid(p)
        return p,normalized_tensor

    def propagate_train(self,graph,h,t,p):
        
        samples = (np.random.rand(len(p)) < np.array(p.cpu())).astype(int)                                                                                           
        act_h=h[samples.nonzero()] 
        act_t=t[samples.nonzero()] 
        value= [1]*len(act_h)

        # Create a COO matrix to represent the newly activated edges
        new_edges_data = np.array(value)
        new_edges = coo_matrix((new_edges_data, (np.array(act_t.cpu()), np.array(act_h.cpu()))), shape=(self.n_users, self.n_users))
        new_graph = graph + new_edges
        
        return list(set(act_t.tolist())), new_graph

    def propagate_inference(self,graph,h,t,p):
        
        samples = (np.array(p.cpu()) > 0.5).astype(int)  
        act_h=h[samples.nonzero()]
        act_t=t[samples.nonzero()]
        value= [1]*len(act_h)

        # Create a COO matrix to represent the newly activated edges
        new_edges_data = np.array(value)
        new_edges = coo_matrix((new_edges_data, (np.array(act_t.cpu()), np.array(act_h.cpu()))), shape=(self.n_users, self.n_users))
        new_graph = graph + new_edges
        
        return list(set(act_t.tolist())), new_graph

    def generate_diffusion_view_inference(self):
         
        graph = coo_matrix((self.n_users, self.n_users), dtype=int)

        ## choose initial seed
        initial_seed  = self.data.kol_idx
        new_active = initial_seed[:]

        ## start diffusion, terminates when there is no activation
        with torch.no_grad():
            for i in range(self.max_iterations):
                if len(new_active)==0:
                    break
                # probability of activating neighbor
                h,t=self.get_head_tail(graph,new_active)
                _,p= self.infer_activation(h,t)
                #get the neighbor being activated and update the probability to the graph
                new_active,graph= self.propagate_inference(graph,h,t,p)
    
        weighted_graph= self.get_weighted_graph(graph)

        return weighted_graph

    def generate_diffusion_view_train(self):

        #start from an empty graph
        graph = coo_matrix((self.n_users, self.n_users), dtype=int)
        ## choose initial seed, sample based on page-rank
        initial_seed = np.random.choice(self.data.n_social, size=int(self.data.n_social*self.act_ratio), replace=False, p=self.data.importance_score)
        new_active = self.data.social_user[initial_seed]

        ## start diffusion, terminates when there is no activation
        with torch.no_grad():
            for i in range(self.max_iterations):
                if len(new_active)==0:
                    break
                # probability of activating neighbor
                h,t=self.get_head_tail(graph,new_active)
                _,p= self.infer_activation(h,t)
                #get the neighbor being activated and update the probability to the graph
                new_active,graph= self.propagate_train(graph,h,t,p)
    
        weighted_graph= self.get_weighted_graph(graph)

        return weighted_graph
    
    
    def aux_hier_cl(self,batch_user,user_embedding):

        ua=[]
        ub_kol=[]
        normal_a=[]
        normal_b=[]
        ub_unobserved=[]

        #SAMPLE CONTRASTIVE PAIRS
        for i in batch_user:
            if i.item() in self.data.cl_dict.keys():

                ua.extend(self.data.cl_dict[i.item()][0])
                ub_kol.extend(self.data.cl_dict[i.item()][1])
                normal_a.extend(self.data.cl_dict[i.item()][2])
                normal_b.extend(self.data.cl_dict[i.item()][3])
                ub_unobserved.extend(self.data.cl_dict[i.item()][4])

        ua=torch.tensor(ua,dtype=torch.int32).to(self.config.device)
        ub_kol=torch.tensor(ub_kol,dtype=torch.int32).to(self.config.device)
        normal_a=torch.tensor(normal_a,dtype=torch.int64).to(self.config.device)
        normal_b=torch.tensor(normal_b,dtype=torch.int64).to(self.config.device)
        ub_unobserved=torch.tensor(ub_unobserved,dtype=torch.int32).to(self.config.device)

        activation1 = self.infer_activation(ub_kol,ua)[0] - self.infer_activation(normal_b,normal_a)[0]
        activation2 = self.infer_activation(normal_b,normal_a)[0]-self.infer_activation(ub_unobserved,normal_a)[0]

        weight = F.cosine_similarity(user_embedding[normal_a,:] , user_embedding[normal_b,:],dim=1)

        margin1=temp_sigmoid(-weight,self.sig_temp,self.yita) # HIGH SIM LOW MARGIN
        margin2=temp_sigmoid(weight,self.sig_temp,self.yita)  # HIGH SIM HIGH MARGIN
        loss1 = F.relu(-activation1+ margin1)
        loss2 = F.relu(-activation2+ margin2)
        loss =  (torch.sum(loss1,dim=0)+torch.sum(loss2,dim=0))/float((len(activation1)))

        return loss    
    
    def bpr_loss(self,users,pos_idx,neg_idx):

        unique_user=torch.tensor(list(set(users.tolist()))).to(self.config.device)

        ## U-I
        inter_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight],0)
        inter_light_out,_ = self.ui_encoder(self.norm_inter_graph,inter_emb)
        user_embed, item_embed = torch.split(inter_light_out, [self.n_users, self.n_items])
    
        ##information bottleneck
        aug=self.generate_diffusion_view_train()

        #original view
        social_embed1,_ = self.uu_encoder1(self.norm_social_graph,user_embed) 
        norm_social_embed1 = F.normalize(social_embed1, dim=1)
        #denoised view
        social_embed2,_ = self.uu_encoder2(aug,user_embed)
        norm_social_embed2 = F.normalize(social_embed2, dim=1)
        final_user_embed=user_embed+social_embed2

        batch_u1=norm_social_embed1[unique_user,:]
        batch_u2=norm_social_embed2[unique_user,:]
        pos_ratings_user = inner_product(batch_u1, batch_u2)    # [batch_size]
        tot_ratings_user1 = torch.matmul(batch_u2, 
                                        torch.transpose(norm_social_embed1, 0, 1))
        #minimize mutual information 
        ssl_loss1 = torch.log(torch.exp(pos_ratings_user/self.ssl_temp) / torch.exp(tot_ratings_user1/self.ssl_temp).sum(dim=1, keepdim=False)).mean()

        # maximize downstream performance
        pos_item_emb = item_embed[pos_idx,:]
        neg_item_emb = item_embed[neg_idx,:]
        batch_user_emb = final_user_embed[users,:]

        pos_scores = torch.mul(batch_user_emb, pos_item_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(batch_user_emb, neg_item_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        bpr_loss=torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        ### auxiliary contrastive loss
        ssl_loss2=self.aux_hier_cl(unique_user,user_embed)

        ##  regularization
        user0=self.user_embedding(unique_user)
        pos0=self.item_embedding(pos_idx)
        neg0=self.item_embedding(neg_idx)
        reg_loss = (1/2)*(user0.norm(2).pow(2)+ \
                  pos0.norm(2).pow(2)+ neg0.norm(2).pow(2))/float(len(unique_user))

        ##entropy regularization
        h=torch.tensor(self.data.social_data[1],dtype=torch.int32).to(self.config.device)
        t=torch.tensor(self.data.social_data[0],dtype=torch.int32).to(self.config.device)
        _,value=self.infer_activation(h,t)
        probabilities = F.softmax(value,dim=0)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1)

        return bpr_loss,reg_loss,ssl_loss1,ssl_loss2,entropy
    
    def forward(self,user_id,item_id):

        ## U-I
        inter_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight],0)
        inter_light_out,_ = self.ui_encoder(self.norm_inter_graph,inter_emb)
        user_embed, item_emb = torch.split(inter_light_out, [self.n_users, self.n_items])
    
        w_graph=self.generate_diffusion_view_inference()
        social_embed,_ = self.uu_encoder2(w_graph,user_embed)
        final_user_embed=user_embed+social_embed

        cf_scores = torch.mul(final_user_embed[user_id,:], item_emb[item_id,:])
        cf_scores = torch.sum(cf_scores, dim=1)

        return cf_scores

    def getUsersRating(self,users):

        ## U-I
        inter_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight],0)
        inter_light_out,_ = self.ui_encoder(self.norm_inter_graph,inter_emb)
        user_embed, item_emb = torch.split(inter_light_out, [self.n_users, self.n_items])
    
        w_graph=self.generate_diffusion_view_inference()
        social_embed,_ = self.uu_encoder2(w_graph,user_embed)
      
        final_user_embed=user_embed+social_embed
        cf_scores = torch.mm(final_user_embed[users,:], item_emb.t())

        return cf_scores

    
        



         