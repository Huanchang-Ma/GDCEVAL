import sys
import os

# Assuming the parent directory of the 'utils' folder is one level up from the current directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import math
import time
from parsers.config import get_config
import logging
from tqdm import tqdm
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader

from utils_reGen.models.gnn import GNN
from config.arguments import load_arguments_from_yaml, get_args
from dataset.get_datasets import get_dataset
from utils_reGen.misc import AverageMeter, validate, print_info, init_weights, ImbalancedSampler
from utils_reGen.solver import load_generator
from augment import build_test_ngds

from utils_reGen.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
                         load_ema, load_loss_fn, load_batch

cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
reg_criterion = torch.nn.MSELoss(reduction='none')


def get_logger(name, logfile=None):
    """ create a nice logger """
    logger = logging.getLogger(name)
    # clear handlers if they were created in other runs
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # create console handler add add to logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create file handler add add to logger when name is not None
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


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


def train(args, model, train_loaders, optimizer, scheduler, epoch):
    if args.task_type in 'regression':
        criterion = reg_criterion
    else:
        criterion = cls_criterion
    if not args.no_print:
        p_bar = tqdm(range(args.steps))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    device = args.device
    model.train()
    for batch_idx in range(args.steps):
        end = time.time()
        model.zero_grad()
        try:
            batch_labeled = next(train_loaders['labeled_iter'])
        except:
            train_loaders['labeled_iter'] = iter(train_loaders['labeled_trainloader'])
            batch_labeled = next(train_loaders['labeled_iter'])
        batch_labeled = batch_labeled.to(device)
        targets = batch_labeled.y.to(torch.float32)
        is_labeled = targets == targets
        if batch_labeled.x.shape[0] == 1 or batch_labeled.batch[-1] == 0:
            continue
        else:
            pred_labeled = model(batch_labeled)[0]
            Losses = criterion(pred_labeled.view(targets.size()).to(torch.float32)[is_labeled], targets[is_labeled])
            loss = Losses.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        if not args.no_print:
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.8f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.steps,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                ))
            p_bar.update()
    if not args.no_print:
        p_bar.close()
    return train_loaders


def main(args):
    device = torch.device('cuda', args.gpu_id)
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    args.n_gpu = torch.cuda.device_count()
    args.device = device
    labeled_dataset = get_dataset(args, './raw_data')
    label_split_idx = labeled_dataset.get_idx_split()
    args.num_trained = len(label_split_idx["train"])
    args.num_trained_init = args.num_trained
    args.task_type = labeled_dataset.task_type
    args.steps = args.num_trained // args.batch_size + 1
    args.strategy = args.strategy_init

    if args.dataset == 'ogbg-molhiv':
        sampler = ImbalancedSampler(labeled_dataset, label_split_idx["train"])
        labeled_trainloader = DataLoader(labeled_dataset[label_split_idx["train"]], batch_size=args.batch_size,
                                         sampler=sampler, num_workers=args.num_workers)
    else:
        labeled_trainloader = DataLoader(labeled_dataset[label_split_idx["train"]], batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.num_workers)

    valid_loader = DataLoader(labeled_dataset[label_split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(labeled_dataset[label_split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    print(len(labeled_trainloader), len(valid_loader), len(test_loader))
    print(len(label_split_idx["valid"]), label_split_idx["valid"], 'label_split_idx["valid"]')
    tensor_list = label_split_idx["valid"].tolist()

    # Save list as JSON
    import json
    with open(os.path.join(args.data_dir, f'val_idx_{args.dataset.lower()}.json'), 'w') as f:
        json.dump(tensor_list, f)
    print(len(label_split_idx["test"]), label_split_idx["test"], 'label_split_idx["test"]')
    assert False


    #_ = load_seed(args.seed)
    #train_loader, test_loader = load_data(args)
    #for batch_idx, data in enumerate(test_loader):
    #    print(f'TEST: Batch {batch_idx}: data shape = {data}')
    #    if batch_idx == 1:  # Just to observe a couple of batches
    #        break
    #assert False
    #print(next(iter(test_loader)), 'next(iter(test_loader))')
    in_edge_dim = next(iter(test_loader)).edge_attr.shape[1]
    in_feat_dim = next(iter(test_loader)).x.shape[1]
    if args.num_tasks is None:
        num_tasks = next(iter(test_loader)).y.shape[1]
    else:
        num_tasks = args.num_tasks

    model = GNN(gnn_type=args.model, num_tasks=num_tasks, num_layer=args.num_layer,
                emb_dim=args.emb_dim,
                drop_ratio=args.drop_ratio, graph_pooling=args.readout, norm_layer=args.norm_layer).to(device)
    print(model)
    #assert False

    #generator = load_generator(device, path='checkpoints/ogbg-molbace-xin-sample.pkl')
    init_weights(model, args.initw_name, init_gain=0.02)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 100)
    logging.warning(f"device: {args.device}, " f"n_gpu: {args.n_gpu}, ")
    logger.info(dict(args._get_kwargs()))
    logger.info("***** Running training *****")
    logger.info(
        f"  Task = {args.dataset}@{args.num_trained}/{len(label_split_idx['valid'])}/{len(label_split_idx['test'])}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Total train batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.epochs * args.steps}")
    train_loaders = {'labeled_iter': iter(labeled_trainloader), 'labeled_trainloader': labeled_trainloader}
    for epoch in range(0, args.epochs):
        train_loaders = train(args, model, train_loaders, optimizer, scheduler, epoch)
        train_perf = validate(args, model, labeled_trainloader)
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
            #torch.save(model.state_dict(), os.path.join(args.save_pretrain_dir,
            #                                            f'{args.model}_{args.dataset}_model.pt'))
            if not args.no_print:
                print_info('Train', train_perf)
                print_info('Valid', valid_perf)
                print_info('Test', test_perf)
        else:
            # not update
            if not args.no_print:
                print_info('Train', train_perf)
                print_info('Valid', valid_perf)
            if epoch > 30:
                cnt_wait += 1
                if cnt_wait > args.patience:
                    break
    print('Finished training! Best validation results from epoch {}.'.format(best_epoch))
    print_info('train', best_train_perf)
    print_info('valid', best_valid_perf)
    print_info('test', test_perf)
    return best_train_perf, best_valid_perf, test_perf


if __name__ == '__main__':

    print("Begin!")
    args = get_args()

    config = load_arguments_from_yaml(f'./config/GDCEval_{args.dataset}.yaml')
    for arg, value in config.items():
        setattr(args, arg, value)
    #print('config', config)

    args.strategy_init = args.strategy
    datetime_now = datetime.now().strftime("%Y%m%d.%H%M%S")
    logger = get_logger(__name__, logfile=None)
    print(args)
    results = {}

    for exp_num in range(args.trails):
        seed_torch(exp_num)
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
            print('{}: {:.4f}+-{:.4f} {}'.format(mode, np.mean(nums), np.std(nums), nums))