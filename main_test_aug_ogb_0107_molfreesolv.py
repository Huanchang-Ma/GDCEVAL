import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import logging
import random
import argparse
import datetime
import numpy as np
import torch
from ogb.graphproppred import Evaluator
from utils_reGen.models.gnn import GNN
from utils_reGen.misc import _eval_rocauc
from utils_reGen.solver import load_generator
from augment_0107_molfreesolv import build_test_ngds

from utils_reGen.loader import load_seed, load_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score


def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def main(args):
    device = torch.device(args.device)
    _ = load_seed(args.seed)
    _, test_loader, _ = load_data(args)

    in_edge_dim = next(iter(test_loader)).edge_attr.shape[1]
    in_feat_dim = next(iter(test_loader)).x.shape[1]
    if args.num_tasks is None:
        num_tasks = next(iter(test_loader)).y.shape[1]
    else:
        num_tasks = args.num_tasks

    model = GNN(gnn_type=args.model, num_tasks=num_tasks, num_layer=args.num_layer, in_edge_dim=in_edge_dim,
                in_feat_dim=in_feat_dim,
                emb_dim=args.emb_dim,
                drop_ratio=args.drop_ratio, graph_pooling=args.readout, norm_layer=args.norm_layer).to(device)
    # model.load_state_dict(torch.load(args.pretrain_model_path))
    model.load_state_dict(torch.load(args.pretrain_model_path), strict=False)

    # Don't forget to set the correct mode for the model
    model.eval()  # For inference
    generator = load_generator(device, path=f'checkpoints/{args.dataset}/{args.ckpt}')
    y_ground_true, y_pred_test_aug, y_pred_test_aug_avg, y_pred_test_ori = build_test_ngds(args, test_loader, model, generator)
    # y_ground_true, y_pred_test_aug, y_pred_test_aug_avg, y_pred_test_ori, y_pred_test_aug_binary, y_pred_test_ori_binary = build_test_ngds(args, test_loader, model, generator)

    if args.task_type == 'regression':
        rmse_aug = mean_squared_error(y_ground_true, y_pred_test_aug, squared=False)
        rmse_aug_avg = mean_squared_error(y_ground_true, y_pred_test_aug_avg, squared=False)
        rmse_ori = mean_squared_error(y_ground_true, y_pred_test_ori, squared=False)
        perf = {'rmse_aug': rmse_aug, 'rmse_aug_avg': rmse_aug_avg, 'rmse_ori': rmse_ori}
        logging.info('Perf RMSE = {} '.format(perf))
        with open('output_samples.txt', 'a') as file_samples:
            file_samples.write(f"perf auc：{perf}\n")
        file_samples.close()

    elif args.task_type == 'classification':
        if args.dataset.startswith('ogbg'):
            evaluator = Evaluator(name=args.dataset)
            auc_aug = evaluator.eval({"y_true": y_ground_true, "y_pred": y_pred_test_aug})['rocauc']
            auc_aug_avg = evaluator.eval({"y_true": y_ground_true, "y_pred": y_pred_test_aug_avg})['rocauc']
            auc_ori = evaluator.eval({"y_true": y_ground_true, "y_pred": y_pred_test_ori})['rocauc']
            # auc_aug2 = roc_auc_score(y_ground_true, y_pred_test_aug_binary)
            # auc_ori2 = roc_auc_score(y_ground_true, y_pred_test_ori_binary)
            # auc_ori_aug = roc_auc_score(y_pred_test_ori, y_pred_test_aug)
        else:
            assert y_ground_true.shape[1] == y_pred_test_aug.shape[1]
            auc_aug = _eval_rocauc(y_ground_true, y_pred_test_aug)['rocauc']
            auc_aug_avg = _eval_rocauc(y_ground_true, y_pred_test_aug_avg)['rocauc']
            auc_ori = _eval_rocauc(y_ground_true, y_pred_test_ori)['rocauc']
            # auc_ori_aug = roc_auc_score(y_pred_test_ori, y_pred_test_aug)
        perf = {'auc_aug': auc_aug, 'auc_aug_avg': auc_aug_avg, 'auc_ori': auc_ori}
        # perf = {'auc_aug': auc_aug, 'auc_aug_avg': auc_aug_avg, 'auc_ori': auc_ori, 'auc_ori_aug': auc_ori_aug}
        logging.info('Perf AUC = {}'.format(perf))
        with open('output_samples.txt', 'a') as file_samples:
            file_samples.write(f"auc_ori：{auc_ori}\n")
            file_samples.write(f"auc_aug：{auc_aug}\n")
            file_samples.write(f"auc_aug_avg：{auc_aug_avg}\n")
            # file_samples.write(f"auc_ori2：{auc_ori2}\n")
            # file_samples.write(f"auc_aug2：{auc_aug2}\n")
        file_samples.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default="ogbg-molbace", type=str,         ## ogbg-molbace
    #                     choices=['plym-density', 'plym-oxygen', 'plym-melting', 'plym-glass',
    #                              'ogbg-mollipo', 'ogbg-molfreesolv', 'ogbg-molesol',
    #                              'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molclintox', 'ogbg-molsider',
    #                              'ogbg-moltox21', 'ogbg-moltoxcast'],
    #                     help='dataset name (plym-, ogbg-)')
    # parser.add_argument('--dataset', default="ogbg-molbbbp", type=str,         ## ogbg-molbbbp
    #                     choices=['plym-density', 'plym-oxygen', 'plym-melting', 'plym-glass',
    #                              'ogbg-mollipo', 'ogbg-molfreesolv', 'ogbg-molesol',
    #                              'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molclintox', 'ogbg-molsider',
    #                              'ogbg-moltox21', 'ogbg-moltoxcast'],
    #                     help='dataset name (plym-, ogbg-)')
    # parser.add_argument('--dataset', default="ogbg-molclintox", type=str,  ## ogbg-molclintox
    #                     choices=['plym-density', 'plym-oxygen', 'plym-melting', 'plym-glass',
    #                              'ogbg-mollipo', 'ogbg-molfreesolv', 'ogbg-molesol',
    #                              'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molclintox', 'ogbg-molsider',
    #                              'ogbg-moltox21', 'ogbg-moltoxcast'],
    #                     help='dataset name (plym-, ogbg-)')
    parser.add_argument('--dataset', default="ogbg-molfreesolv", type=str,  ## ogbg-molfreesolv
                        choices=['plym-density', 'plym-oxygen', 'plym-melting', 'plym-glass',
                                 'ogbg-mollipo', 'ogbg-molfreesolv', 'ogbg-molesol',
                                 'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molclintox', 'ogbg-molsider',
                                 'ogbg-moltox21', 'ogbg-moltoxcast'],
                        help='dataset name (plym-, ogbg-)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--task_type', type=str, default='classification', choices=['classification', 'regression'])    ## ogbg-molbace ogbg-molbbbp ogbg-molclintox
    parser.add_argument('--task_type', type=str, default='regression', choices=['classification', 'regression'])    ## ogbg-molfreesolv
    parser.add_argument('--num_tasks', type=int, default=1)    ##'ogbg-molbace', 'ogbg-molbbbp' 'ogbg-molfreesolv'
    # parser.add_argument('--num_tasks', type=int, default=2)    ##'ogbg-molclintox'


    # model
    # parser.add_argument('--model', type=str, default='gin',
    #                     help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    # parser.add_argument('--model', type=str, default='gcn',
    #                     help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    # parser.add_argument('--model', type=str, default='gat',
    #                     help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--model', type=str, default='sage',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--readout', type=str, default='sum',
                        help='graph readout (default: sum)')
    parser.add_argument('--norm_layer', type=str, default='batch_norm',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    ## model GIN
    # parser.add_argument('--pretrain_model_path', type=str,              ## ogbg-molbace
    #                     default='pretrain_models/ogbg-molbace-gin-20240804-154912-964894/gin_ogbg-molbace_model.pt',
    #                     help='pretrained gin model on the training set')
    # parser.add_argument('--pretrain_model_path', type=str,           ## ogbg-molbbbp
    #                     default='pretrain_models/ogbg-molbbbp-gin-20240829-161648-933582/gin_ogbg-molbbbp_model.pt',
    #                     help='pretrained gin model on the training set')
    # parser.add_argument('--pretrain_model_path', type=str,  ## ogbg-molclintox
    #                     default='pretrain_models/ogbg-molclintox-gin-20240920-114540-263439/gin_ogbg-molclintox_model.pt',
    #                     help='pretrained gin model on the training set')
    # parser.add_argument('--pretrain_model_path', type=str,  ## ogbg-molfreesolv
    #                     default='pretrain_models/ogbg-molfreesolv-gin-20240818-200101-328916/gin_ogbg-molfreesolv_model.pt',
    #                     help='pretrained gin model on the training set')
    ## model GCN
    # parser.add_argument('--pretrain_model_path', type=str,  ## ogbg-molbace
    #                     default='pretrain_models/ogbg-molbace-gcn-20241210-174722-453823/gcn_ogbg-molbace_model.pt',
    #                     help='pretrained gcn model on the training set')
    # parser.add_argument('--pretrain_model_path', type=str,           ## ogbg-molbbbp
    #                     default='pretrain_models/ogbg-molbbbp-gcn-20241210-174820-979525/gcn_ogbg-molbbbp_model.pt',
    #                     help='pretrained gcn model on the training set')
    # parser.add_argument('--pretrain_model_path', type=str,  ## ogbg-molclintox
    #                     default='pretrain_models/ogbg-molclintox-gcn-20241210-194917-667224/gcn_ogbg-molclintox_model.pt',
    #                     help='pretrained gcn model on the training set')
    # parser.add_argument('--pretrain_model_path', type=str,  ## ogbg-molfreesolv
    #                     default='pretrain_models/ogbg-molfreesolv-gcn-20241210-203836-285073/gcn_ogbg-molfreesolv_model.pt',
    #                     help='pretrained gcn model on the training set')
    ## model GAT
    # parser.add_argument('--pretrain_model_path', type=str,  ## ogbg-molbace
    #                     default='pretrain_models/ogbg-molbace-gat-20241228-151706-961596/gat_ogbg-molbace_model.pt',
    #                     help='pretrained gcn model on the training set')
    # parser.add_argument('--pretrain_model_path', type=str,           ## ogbg-molbbbp
    #                     default='pretrain_models/ogbg-molbbbp-gat-20241228-151855-331042/gat_ogbg-molbbbp_model.pt',
    #                     help='pretrained gcn model on the training set')
    # parser.add_argument('--pretrain_model_path', type=str,  ## ogbg-molclintox
    #                     default='pretrain_models/ogbg-molclintox-gat-20241228-152311-841936/gat_ogbg-molclintox_model.pt',
    #                     help='pretrained gcn model on the training set')
    # parser.add_argument('--pretrain_model_path', type=str,  ## ogbg-molfreesolv
    #                     default='pretrain_models/ogbg-molfreesolv-gat-20241228-151433-180090/gat_ogbg-molfreesolv_model.pt',
    #                     help='pretrained gcn model on the training set')
    ## model GraphSAGE
    # parser.add_argument('--pretrain_model_path', type=str,  ## ogbg-molbace
    #                     default='pretrain_models/ogbg-molbace-sage-20241228-151625-273773/sage_ogbg-molbace_model.pt',
    #                     help='pretrained gcn model on the training set')
    # parser.add_argument('--pretrain_model_path', type=str,           ## ogbg-molbbbp
    #                     default='pretrain_models/ogbg-molbbbp-sage-20241228-152002-221407/sage_ogbg-molbbbp_model.pt',
    #                     help='pretrained gcn model on the training set')
    # parser.add_argument('--pretrain_model_path', type=str,  ## ogbg-molclintox
    #                     default='pretrain_models/ogbg-molclintox-sage-20241228-152140-880041/sage_ogbg-molclintox_model.pt',
    #                     help='pretrained gcn model on the training set')
    parser.add_argument('--pretrain_model_path', type=str,  ## ogbg-molfreesolv
                        default='pretrain_models/ogbg-molfreesolv-sage-20241228-151524-197344/sage_ogbg-molfreesolv_model.pt',
                        help='pretrained gcn model on the training set')
    parser.add_argument('--sample_steps', nargs='+', type=int, default=[3, 5, 7, 10, 15], help='List of sample steps')
    # parser.add_argument('--p_ratio_lists', nargs='+', type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9],
    #                     help='List of sample p_ratio_lists')
    parser.add_argument('--p_ratio_lists', nargs='+', type=float, default=[0.1],
                        help='List of sample p_ratio_lists')

    # training
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 256)')
    # parser.add_argument('--ckpt', type=str, default='Aug02-16_17_12.pth',    ## ogbg-molbace
    #                     help='diffusion ckpt path')
    # parser.add_argument('--ckpt', type=str, default='Aug21-13_05_16.pth',    ## ogbg-molbbbp
    #                     help='diffusion ckpt path')
    # parser.add_argument('--ckpt', type=str, default='Aug30-08_20_13.pth',  ## ogbg-molclintox
    #                     help='diffusion ckpt path')
    parser.add_argument('--ckpt', type=str, default='Aug06-10_08_17.pth',  ## ogbg-molfreesolv
                        help='diffusion ckpt path')
    # augmentation
    parser.add_argument('--snr', type=float, default=0.2,
                        help='snr')
    parser.add_argument('--scale_eps', type=float, default=0,
                        help='scale eps')
    parser.add_argument('--te_aug_ratio', type=float, default=1,
                        help='ratio of embedding combination')
    parser.add_argument('--save_log', type=str, default='diff_results_xz',
                        help="method to initialize the model paramter")

    parser.add_argument('--ly_w', type=float, default=1,
                        help="weight for ly_w * loss_y")
    parser.add_argument('--lgemb_w', type=float, default=0,
                        help="weight for lgemb_w * loss_g_embs")
    parser.add_argument('--lfgw_w', type=float, default=1,
                        help="weight for loss_fgw_dist")
    parser.add_argument('--lstrut_w', type=float, default=1,
                        help="weight for lstru_w * loss_struct")
    args = parser.parse_args()

    ## n_steps for solver
    args.n_steps = 1
    device = torch.device(args.device)
    if not os.path.exists(args.save_log):
        os.makedirs(args.save_log)
    args.timestamp = datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f")
    args.data_dir = 'data'
    log_dir = args.save_log + '/diff-{}-{}-{}'.format(args.dataset, str(args.model),
                                                      args.timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.log_dir = log_dir
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'diffusion_aug.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('This is the log_dir: {}'.format(log_dir))
    logging.info('This is the param space: {}'.format(args))
    main(args)

    logging.info('Finish! Log_dir final is: {}'.format(log_dir))
