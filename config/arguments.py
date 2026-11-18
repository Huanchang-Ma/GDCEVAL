import yaml
import argparse
import sys
import os

# Assuming the parent directory of the 'utils' folder is one level up from the current directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
def load_arguments_from_yaml(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_args():
    parser = argparse.ArgumentParser(description='Data-Centric Learning from Unlabeled Graphs with Diffusion Model')
    #parser.add_argument('--gpu-id', type=int, default=0,
    #                    help='which gpu to use if any (default: 0)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers for data loader')
    parser.add_argument('--no-print', action='store_true', default=False,
                        help="don't use progress bar")
    # parser.add_argument('--dataset', default="ogbg-molbace", type=str,
    #                     choices=['plym-density', 'plym-oxygen', 'plym-melting', 'plym-glass',
    #                             'ogbg-mollipo', 'ogbg-molfreesolv', 'ogbg-molesol',
    #                             'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molclintox','ogbg-molsider','ogbg-moltox21','ogbg-moltoxcast'],
    #                     help='dataset name (plym-, ogbg-)')
    # parser.add_argument('--dataset', default="ogbg-molbbbp", type=str,
    #                     choices=['plym-density', 'plym-oxygen', 'plym-melting', 'plym-glass',
    #                              'ogbg-mollipo', 'ogbg-molfreesolv', 'ogbg-molesol',
    #                              'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molclintox', 'ogbg-molsider',
    #                              'ogbg-moltox21', 'ogbg-moltoxcast'],
    #                     help='dataset name (plym-, ogbg-)')
    parser.add_argument('--dataset', default="ogbg-molclintox", type=str,
                        choices=['plym-density', 'plym-oxygen', 'plym-melting', 'plym-glass',
                                 'ogbg-mollipo', 'ogbg-molfreesolv', 'ogbg-molesol',
                                 'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molclintox', 'ogbg-molsider',
                                 'ogbg-moltox21', 'ogbg-moltoxcast'],
                        help='dataset name (plym-, ogbg-)')

    # model
    parser.add_argument('--model', type=str, default='gin',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--readout', type=str, default='sum',
                        help='graph readout (default: sum)')
    parser.add_argument('--norm-layer', type=str, default='batch_norm', 
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop-ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num-layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb-dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    # training
    parser.add_argument('--batch-size', type=int, default=512,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for early stop')
    parser.add_argument('--trails', type=int, default=5,
                        help='nubmer of experiments (default: 5)')   
    parser.add_argument('--lr', '--learning-rate', type=float, default=1e-2,
                        help='Learning rate (default: 1e-2)')
    parser.add_argument('--wdecay', default=1e-5, type=float,
                        help='weight decay')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument('--initw_name', type=str, default='xavier',
                        help="method to initialize the model paramter")
    parser.add_argument('--use_scheduler', type=int, default=0,
                        help="use cosine schedular or not", choices=[0, 1])
    parser.add_argument('--task_type', type=str, default='classification',
                        help="molecular task type", choices=['classification','regression'])
    parser.add_argument('--save_log', type=str, default='pretrain_models',
                        help="save training log")



    args = parser.parse_args()
    print('no print',args.no_print)

    ## n_steps for solver
    args.n_steps = 1
    return args