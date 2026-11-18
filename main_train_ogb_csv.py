import sys
import os
# Assuming the parent directory of the 'utils' folder is one level up from the current directory
#parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#sys.path.append(parent_dir)
import math
import argparse
import time
from parsers.config import get_config
import logging
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader

from utils_reGen.models.gnn import GNN
from config.arguments import load_arguments_from_yaml, get_args
from utils_reGen.misc import AverageMeter, validate, print_info, init_weights
from utils_reGen.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
    load_ema, load_loss_fn, load_batch

cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
reg_criterion = torch.nn.MSELoss(reduction='none')


def seed_torch(seed=0):
    print('Seed', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=0.5,
                                    # num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(1e-2, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def train(args, model, train_loaders, optimizer, scheduler, epoch, device):
    if args.task_type in 'regression':
        criterion = reg_criterion
    else:
        criterion = cls_criterion
    losses = AverageMeter()
    model.train()
    for batch_idx, train_b in enumerate(train_loaders):
        train_b = train_b.to(device)
        model.zero_grad()
        pred_labeled = model(train_b)[0]
        Losses = criterion(pred_labeled.to(torch.float32),
                           train_b.y)
        loss = Losses.mean()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.update(loss.item())

    return losses.avg


def main(args):
    device = torch.device(args.device)
    _ = load_seed(args.seed)
    train_loader, test_loader, valid_loader = load_data(args)
    # assert False

    in_edge_dim = next(iter(train_loader)).edge_attr.shape[1]
    in_feat_dim = next(iter(train_loader)).x.shape[1]
    if args.num_tasks is None:
        num_tasks = next(iter(train_loader)).y.shape[1]
    else:
        num_tasks = args.num_tasks

    model = GNN(gnn_type=args.model, num_tasks=num_tasks, num_layer=args.num_layer, in_edge_dim=in_edge_dim,
                in_feat_dim=in_feat_dim,
                emb_dim=args.emb_dim,
                drop_ratio=args.drop_ratio, graph_pooling=args.readout, norm_layer=args.norm_layer).to(device)

    init_weights(model, args.initw_name, init_gain=0.02)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.use_scheduler == 1:
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 100)
    else:
        scheduler = None
    logging.info("***** Running training *****")
    logging.info("  Task = {} | Num Epochs = {}".format(args.dataset, args.epochs))

    for epoch in range(0, args.epochs):
        loss_train = train(args, model, train_loader, optimizer, scheduler, epoch, device)
        if not args.no_print and scheduler is not None:
            if epoch % 20 == 0:
                logging.info("  Train Epoch = {} | LR = {} | Loss = {}".format(
                    epoch,
                    round(scheduler.get_last_lr()[0], 6),
                    round(loss_train, 4)
                ))
        else:
            if epoch % 20 == 0:
                logging.info("  Train Epoch = {} | LR = {} | Loss = {}".format(
                    epoch,
                    round(args.lr, 6),
                    round(loss_train, 4)
                ))

        train_perf = validate(args, model, train_loader)
        valid_perf = validate(args, model, valid_loader)

        update_test = False
        if epoch != 0 and 'classification' in args.task_type and valid_perf['auc'] > best_valid_perf['auc']:
            update_test = True
        elif epoch != 0 and 'regression' in args.task_type and valid_perf['mae'] < best_valid_perf['mae']:
            update_test = True
        if update_test or epoch == 0:
            best_valid_perf = valid_perf
            best_train_perf = train_perf
            cnt_wait = 0
            best_epoch = epoch
            test_perf = validate(args, model, test_loader)
            torch.save(model.state_dict(), os.path.join(args.log_dir,
                                                        f'{args.model}_{args.dataset}_model.pt'))
        else:
            if epoch > 30:
                cnt_wait += 1
                if cnt_wait > args.patience:
                    logging.info("Early stopping at epoch {}".format(epoch))
                    break
    logging.info('Finished training!')
    logging.info(
        'Best Epoch = {} | Best Train = {} | Best Valid = {} | Best Test = {}'.format(best_epoch, best_train_perf,
                                                                                      best_valid_perf, test_perf))

    return best_train_perf, best_valid_perf, test_perf


if __name__ == '__main__':

    print("Begin!")
    parser = argparse.ArgumentParser(description='TTGNDS')
    # parser.add_argument('--gpu-id', type=int, default=0,
    #                    help='which gpu to use if any (default: 0)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for data loader')
    parser.add_argument('--no_print', action='store_true', default=False,
                        help="don't use progress bar")
    # parser.add_argument('--dataset', default="ogbg-molbace", type=str,
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
    # parser.add_argument('--dataset', default="ogbg-molbbbp", type=str,
    #                     choices=['plym-density', 'plym-oxygen', 'plym-melting', 'plym-glass',
    #                              'ogbg-mollipo', 'ogbg-molfreesolv', 'ogbg-molesol',
    #                              'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molclintox', 'ogbg-molsider',
    #                              'ogbg-moltox21', 'ogbg-moltoxcast'],
    #                     help='dataset name (plym-, ogbg-)')
    # parser.add_argument('--dataset', default="ogbg-molfreesolv", type=str,
    #                     choices=['plym-density', 'plym-oxygen', 'plym-melting', 'plym-glass',
    #                              'ogbg-mollipo', 'ogbg-molfreesolv', 'ogbg-molesol',
    #                              'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molclintox', 'ogbg-molsider',
    #                              'ogbg-moltox21', 'ogbg-moltoxcast'],
    #                     help='dataset name (plym-, ogbg-)')
    # model
    # parser.add_argument('--model', type=str, default='gin',
    #                     help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    # parser.add_argument('--model', type=str, default='sage',
    #                     help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--model', type=str, default='gat',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--readout', type=str, default='sum',
                        help='graph readout (default: sum)',choices=['sum', 'max','mean'])
    parser.add_argument('--norm_layer', type=str, default='batch_norm',
                        help='normalization layer (default: size_norm)',choices=['size_norm', 'batch_norm'])
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    # training
    parser.add_argument('--batch_size', type=int, default=64,    ## molbace molbbbp molclintox
                        help='input batch size for training (default: 256)')
    # parser.add_argument('--batch_size', type=int, default=48,    ## molfreesolv
    #                     help='input batch size for training (default: 256)')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for early stop')
    parser.add_argument('--trails', type=int, default=1,
                        help='nubmer of experiments (default: 1)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.005,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--wdecay', default=1e-5, type=float,
                        help='weight decay')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train')
    # parser.add_argument('--num_tasks', type=int, default=1,    ## molbace molbbbp molfreesolv
    #                     help='number of task')
    parser.add_argument('--num_tasks', type=int, default=2,    ## molclintox
                        help='number of task')
    parser.add_argument('--initw_name', type=str, default='xavier',
                        help="method to initialize the model paramter")
    parser.add_argument('--use_scheduler', type=int, default=0,
                        help="use cosine schedular or not", choices=[0, 1])
    parser.add_argument('--task_type', type=str, default='classification',
                        help="molecular task type", choices=['classification', 'regression'])
    # parser.add_argument('--task_type', type=str, default='regression',
    #                     help="molecular task type", choices=['classification', 'regression'])  ## ogbg-molfreesolv
    parser.add_argument('--save_log', type=str, default='pretrain_models',
                        help="save training log")
    args = parser.parse_args()
    args.data_dir = 'data/'
    #args = get_args()

    #config = load_arguments_from_yaml(f'config/GIN_{args.dataset}_train.yaml')
    #for arg, value in config.items():
    #    setattr(args, arg, value)

    #datetime_now = datetime.now().strftime("%Y%m%d.%H%M%S")
    log_dir = args.save_log + '/{}-{}-{}'.format(args.dataset, str(args.model),
                                                 datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.log_dir = log_dir
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'pretrain.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('This is the log_dir: {}'.format(log_dir))
    logging.info('This is the param space: {}'.format(args))
    results = {}

    for exp_num in range(args.trails):
        #seed_torch(exp_num)
        args.exp_num = exp_num
        train_perf, valid_perf, test_perf = main(args)
        exp_result_temp = {'train': train_perf, 'valid': valid_perf, 'test': test_perf}
        if exp_num == 0:
            for metric in train_perf.keys():
                results[f'train_{metric}'] = []
                results[f'valid_{metric}'] = []
                results[f'test_{metric}'] = []
        for name in ['train', 'test', 'valid']:
            if args.task_type in 'regression':
                metric_list = ['rmse', 'r2', 'mae', 'mse']
            else:
                metric_list = ['auc']
            for metric in metric_list:
                results[f'{name}_{metric}'].append(exp_result_temp[name][metric])
        for mode, nums in results.items():
            logging.info('{}: {:.4f}+-{:.4f} {}'.format(mode, np.mean(nums), np.std(nums), nums))

    logging.info('Finish! Log_dir final is: {}'.format(log_dir))
