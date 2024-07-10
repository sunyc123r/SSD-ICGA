import argparse
import torch

def parse_SSD_ICGA_args():
    parser = argparse.ArgumentParser(description="Run SSD-ICGA.")
    parser.add_argument('--seed', type=int, default=2019,
                        help='Random seed.')
    parser.add_argument('--data_name', nargs='?', default='Ciao',
                        help='Choose a dataset from {Epinion,Ciao}')
    parser.add_argument('--data_dir', nargs='?', default='datasets/',
                        help='Input data path.')
    parser.add_argument('--train_batch_size', type=int, default=1024,
                        help='batch size.')
    parser.add_argument('--test_batch_size', type=int, default=1024,
                        help='Test batch size')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--inter_layers', type=int, default=3,
                        help='interaction layer number')
    parser.add_argument('--social_layers', type=int, default=3,
                        help='social layer number')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=500,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=50,
                        help='Number of epoch for early stopping')
    parser.add_argument('--print_every', type=int, default=20,
                        help='Iter interval of printing CF loss.')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating CF.')
    parser.add_argument('--Ks', nargs='?', default='[5,10,15,20]',
                        help='Calculate metric@K when evaluating.')
    parser.add_argument('--lambda1', nargs='?', default=0.01,type=float,
                        help='l2 regularization')
    parser.add_argument('--lambda2', nargs='?', default=0.01,type=float,
                        help='ib loss coefficient')
    parser.add_argument('--lambda3', nargs='?', default=0.01,type=float,
                        help='hier cl loss coefficient')
    parser.add_argument('--lambda4', nargs='?', default=0.01,type=float,
                        help='entropy loss coefficient')
    parser.add_argument('--ssl_temp', nargs='?', default=0.3,type=float,
                        help='infonce temperature')
    parser.add_argument('--yita', nargs='?', default=3,type=float,
                        help='sigmoid scale')
    parser.add_argument('--max_iterations', nargs='?', default=5,type=int,
                        help='IC max iteration')
    parser.add_argument('--activation_ratio', nargs='?', default=0.01,type=float,
                        help='initial seed set ratio')
    parser.add_argument('--sig_temp', nargs='?', default=2,type=float,
                        help='sigmoid temp')
    parser.add_argument('--device', nargs='?', default=2,
                        help='device id')
    parser.add_argument('--neg_num', nargs='?', default=1,
                        help='number of negative instance for bpr training')
    args = parser.parse_args()
    #args.device = torch.device("cpu")
    args.device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

    save_dir = 'trained_model/SSD-ICGA/{}/batch_size{}_embed-dim{}_ilayer{}_slayer{}_lr{}_reg{}_ib{}_cl{}_ent{}/'.format(
        args.data_name, args.train_batch_size,args.embed_dim,args.inter_layers,args.social_layers,args.lr,args.lambda1, args.lambda2,args.lambda3,args.lambda4)
    args.save_dir = save_dir

    return args
