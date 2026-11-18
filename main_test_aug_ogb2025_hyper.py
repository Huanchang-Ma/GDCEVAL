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
from augment2025_hyper import build_test_ngds  ## classification
# from augment2024 import build_test_ngds  ## regression

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

    model = GNN(
        gnn_type=args.model,
        num_tasks=num_tasks,
        num_layer=args.num_layer,
        in_edge_dim=in_edge_dim,
        in_feat_dim=in_feat_dim,
        emb_dim=args.emb_dim,
        drop_ratio=args.drop_ratio,
        graph_pooling=args.readout,
        norm_layer=args.norm_layer
    ).to(device)

    # 加载预训练参数
    model.load_state_dict(torch.load(args.pretrain_model_path), strict=False)
    model.eval()

    generator = load_generator(device, path=f'checkpoints/{args.dataset}/{args.ckpt}')

    # 注意：build_test_ngds 里直接用 args.sample_steps / args.p_ratio_lists，
    # 我们在下面多重循环里会对 args 赋不同的值再传进来
    y_ground_true, y_pred_test_aug, y_pred_test_aug_avg, y_pred_test_ori = build_test_ngds(args, test_loader, model, generator)

    if args.task_type == 'regression':
        rmse_aug = np.sqrt(mean_squared_error(y_ground_true, y_pred_test_aug))
        rmse_aug_avg = np.sqrt(mean_squared_error(y_ground_true, y_pred_test_aug_avg))
        rmse_ori = np.sqrt(mean_squared_error(y_ground_true, y_pred_test_ori))
        # rmse_aug = mean_squared_error(y_ground_true, y_pred_test_aug, squared=False)
        # rmse_aug_avg = mean_squared_error(y_ground_true, y_pred_test_aug_avg, squared=False)
        # rmse_ori = mean_squared_error(y_ground_true, y_pred_test_ori, squared=False)
        perf = {
            'rmse_aug': rmse_aug,
            'rmse_aug_avg': rmse_aug_avg,
            'rmse_ori': rmse_ori
        }
        logging.info('Perf RMSE = %s', perf)
        return perf

    elif args.task_type == 'classification':
        if args.dataset.startswith('ogbg'):
            evaluator = Evaluator(name=args.dataset)
            auc_aug = evaluator.eval({"y_true": y_ground_true, "y_pred": y_pred_test_aug})['rocauc']
            auc_aug_avg = evaluator.eval({"y_true": y_ground_true, "y_pred": y_pred_test_aug_avg})['rocauc']
            auc_ori = evaluator.eval({"y_true": y_ground_true, "y_pred": y_pred_test_ori})['rocauc']
        else:
            assert y_ground_true.shape[1] == y_pred_test_aug.shape[1]
            auc_aug = _eval_rocauc(y_ground_true, y_pred_test_aug)['rocauc']
            auc_aug_avg = _eval_rocauc(y_ground_true, y_pred_test_aug_avg)['rocauc']
            auc_ori = _eval_rocauc(y_ground_true, y_pred_test_ori)['rocauc']

        perf = {
            'auc_aug': auc_aug,
            'auc_aug_avg': auc_aug_avg,
            'auc_ori': auc_ori,
        }
        logging.info('Perf AUC = %s', perf)
        return perf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 这些参数作为“基准”，下面会根据不同 dataset / model 覆盖
    parser.add_argument('--dataset', default="ogbg-molbace", type=str,
                        choices=['plym-density', 'plym-oxygen', 'plym-melting', 'plym-glass',
                                 'ogbg-mollipo', 'ogbg-molfreesolv', 'ogbg-molesol',
                                 'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp',
                                 'ogbg-molclintox', 'ogbg-molsider',
                                 'ogbg-moltox21', 'ogbg-moltoxcast'],
                        help='dataset name (plym-, ogbg-)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'regression'])
    parser.add_argument('--num_tasks', type=int, default=1)

    # GNN 模型结构相关
    parser.add_argument('--model', type=str, default='gat',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--readout', type=str, default='sum',
                        help='graph readout (default: sum)')
    parser.add_argument('--norm_layer', type=str, default='batch_norm')
    parser.add_argument('--drop_ratio', type=float, default=0.5)
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--emb_dim', type=int, default=128)

    parser.add_argument('--pretrain_model_path', type=str, default='',
                        help='will be overwritten for each dataset/model')

    # 这里的 list 只是默认值，实际会被网格里的单个值覆盖
    parser.add_argument('--sample_steps', nargs='+', type=int, default=[3, 5, 7, 10, 15],
                        help='List of sample steps')
    parser.add_argument('--p_ratio_lists', nargs='+', type=float, default=[0.1],
                        help='List of sample p_ratio_lists')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ckpt', type=str, default='',
                        help='diffusion ckpt path (will be overwritten)')

    # augmentation
    parser.add_argument('--snr', type=float, default=0.2)
    parser.add_argument('--scale_eps', type=float, default=0.1)
    parser.add_argument('--te_aug_ratio', type=float, default=1)
    parser.add_argument('--save_log', type=str, default='diff_results_xz',
                        help="root dir to save logs and txt results")

    parser.add_argument('--ly_w', type=float, default=1)
    parser.add_argument('--lgemb_w', type=float, default=0)
    parser.add_argument('--lfgw_w', type=float, default=1)
    parser.add_argument('--lstrut_w', type=float, default=1)

    base_args = parser.parse_args()

    import copy

    LOSS_WEIGHT_CFG = {
        # 大一点/稳定一点的数据集：结构约束可以相对强
        # 'ogbg-molfreesolv': dict(
        #     ly_w=1.0,
        #     lgemb_w=0.8,
        #     lfgw_w=1.0,
        #     lstrut_w=1.0,
        # ),
        # 'ogbg-molbace': dict(
        #     ly_w=1.0,
        #     lgemb_w=0.10,
        #     lfgw_w=0.05,
        #     lstrut_w=0.02,
        # ),

        # 小样本/容易过拟合的数据集：结构约束要非常弱
        # 'ogbg-molbbbp': dict(
        #     ly_w=1.0,
        #     lgemb_w=0.05,
        #     lfgw_w=0.01,  # 比上面小一到两个量级
        #     lstrut_w=0.00,  # 先关掉结构 MSE
        # ),
        'ogbg-molbbbp': dict(
            ly_w=1.0,
            lgemb_w=1.0,
            lfgw_w=1.0,
            lstrut_w=0.8
        ),
        # 'ogbg-molclintox': dict(
        #     ly_w=1.0,
        #     lgemb_w=0.05,
        #     lfgw_w=0.02,
        #     lstrut_w=0.00,
        # ),
    }

    # 如果有意外的数据集, 就用一个保守默认值
    # DEFAULT_LOSS_WEIGHTS = dict(
    #     ly_w=1.0,
    #     lgemb_w=0.05,
    #     lfgw_w=0.02,
    #     lstrut_w=0.00,
    # )
    # DEFAULT_LOSS_WEIGHTS = dict(
    #     ly_w=1.0,
    #     lgemb_w=0.00,
    #     lfgw_w=1.0,
    #     lstrut_w=1.0,
    # )
    # DEFAULT_LOSS_WEIGHTS = dict(
    #     ly_w=1.0,
    #     lgemb_w=0.5,
    #     lfgw_w=0.5,
    #     lstrut_w=0.5,
    # )
    DEFAULT_LOSS_WEIGHTS = dict(
        ly_w=1.0,
        lgemb_w=1.0,
        lfgw_w=1.0,
        lstrut_w=0.8,
    )

    def apply_loss_weights(args):
        """
        根据当前 args.dataset 给 args 写入 ly_w, lgemb_w, lfgw_w, lstrut_w
        """
        cfg = LOSS_WEIGHT_CFG.get(args.dataset, DEFAULT_LOSS_WEIGHTS)
        args.ly_w = cfg['ly_w']
        args.lgemb_w = cfg['lgemb_w']
        args.lfgw_w = cfg['lfgw_w']
        args.lstrut_w = cfg['lstrut_w']
        return args

    # =============================
    # 1) 定义要遍历的搜索空间
    # =============================
    # datasets = ['ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molclintox', 'ogbg-molfreesolv']
    datasets = ['ogbg-molbbbp']
    # datasets = ['ogbg-molfreesolv']
    # models = ['gin']
    # datasets = ['ogbg-molclintox']
    # models = ['gat']
    # models = ['gin']
    models = ['gin', 'gcn', 'sage', 'gat']
    sample_steps_candidates = [10]
    p_ratio_candidates = [0.1]
    # models = ['gin']
    # models = ['sage']
    # sample_steps_candidates = [5, 10, 15, 20]
    # p_ratio_candidates = [0.1, 0.2, 0.3, 0.5, 0.7]
    # models = ['gin', 'gcn', 'sage']
    # datasets = ['ogbg-molbace']
    # sample_steps_candidates = [10, 15, 20]
    # p_ratio_candidates = [0.7]
    # models = ['sage']


    # 不同数据集的 task_type / num_tasks
    DATASET_CFG = {
        'ogbg-molbace': dict(task_type='classification', num_tasks=1),
        'ogbg-molbbbp': dict(task_type='classification', num_tasks=1),
        'ogbg-molclintox': dict(task_type='classification', num_tasks=2),
        'ogbg-molfreesolv': dict(task_type='regression', num_tasks=1),
    }

    # 不同模型 + 数据集 对应的预训练权重路径（用你原注释里的路径填）
    PRETRAIN_MODEL_PATHS = {
        'gin': {
            'ogbg-molbace': 'pretrain_models/ogbg-molbace-gin-20240804-154912-964894/gin_ogbg-molbace_model.pt',
            'ogbg-molbbbp': 'pretrain_models/ogbg-molbbbp-gin-20240829-161648-933582/gin_ogbg-molbbbp_model.pt',
            'ogbg-molclintox': 'pretrain_models/ogbg-molclintox-gin-20240920-114540-263439/gin_ogbg-molclintox_model.pt',
            'ogbg-molfreesolv': 'pretrain_models/ogbg-molfreesolv-gin-20240818-200101-328916/gin_ogbg-molfreesolv_model.pt',
        },
        'gcn': {
            'ogbg-molbace': 'pretrain_models/ogbg-molbace-gcn-20241210-174722-453823/gcn_ogbg-molbace_model.pt',
            'ogbg-molbbbp': 'pretrain_models/ogbg-molbbbp-gcn-20241210-174820-979525/gcn_ogbg-molbbbp_model.pt',
            'ogbg-molclintox': 'pretrain_models/ogbg-molclintox-gcn-20241210-194917-667224/gcn_ogbg-molclintox_model.pt',
            'ogbg-molfreesolv': 'pretrain_models/ogbg-molfreesolv-gcn-20241210-203836-285073/gcn_ogbg-molfreesolv_model.pt',
        },
        'gat': {
            'ogbg-molbace': 'pretrain_models/ogbg-molbace-gat-20241228-151706-961596/gat_ogbg-molbace_model.pt',
            'ogbg-molbbbp': 'pretrain_models/ogbg-molbbbp-gat-20241228-151855-331042/gat_ogbg-molbbbp_model.pt',
            'ogbg-molclintox': 'pretrain_models/ogbg-molclintox-gat-20241228-152311-841936/gat_ogbg-molclintox_model.pt',
            'ogbg-molfreesolv': 'pretrain_models/ogbg-molfreesolv-gat-20241228-151433-180090/gat_ogbg-molfreesolv_model.pt',
        },
        'sage': {
            'ogbg-molbace': 'pretrain_models/ogbg-molbace-sage-20241228-151625-273773/sage_ogbg-molbace_model.pt',
            'ogbg-molbbbp': 'pretrain_models/ogbg-molbbbp-sage-20241228-152002-221407/sage_ogbg-molbbbp_model.pt',
            'ogbg-molclintox': 'pretrain_models/ogbg-molclintox-sage-20241228-152140-880041/sage_ogbg-molclintox_model.pt',
            'ogbg-molfreesolv': 'pretrain_models/ogbg-molfreesolv-sage-20241228-151524-197344/sage_ogbg-molfreesolv_model.pt',
        },
    }

    # diffusion checkpoint
    CKPT_PATHS = {
        'ogbg-molbace': 'Aug02-16_17_12.pth',
        'ogbg-molbbbp': 'Aug21-13_05_16.pth',
        'ogbg-molclintox': 'Aug30-08_20_13.pth',
        'ogbg-molfreesolv': 'Aug06-10_08_17.pth',
    }

    # =============================
    # 2) 初始化日志根目录
    # =============================
    # if not os.path.exists(base_args.save_log):
    #     os.makedirs(base_args.save_log)
    #
    # log_format = '%(asctime)s %(message)s'
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    #                     format=log_format, datefmt='%m/%d %I:%M:%S %p')

    # =============================
    # 3) 四重循环跑所有组合
    # =============================
    for ds in datasets:
        for model in models:
            for step in sample_steps_candidates:
                for p_ratio in p_ratio_candidates:
                    args = copy.deepcopy(base_args)
                    args.dataset = ds
                    args.model = model

                    # 数据集相关（分类/回归、任务数量）
                    args.task_type = DATASET_CFG[ds]['task_type']
                    args.num_tasks = DATASET_CFG[ds]['num_tasks']

                    # === 新增：按数据集写入 loss 权重 ===
                    args = apply_loss_weights(args)

                    # 预训练模型 & diffusion ckpt
                    args.pretrain_model_path = PRETRAIN_MODEL_PATHS[model][ds]
                    args.ckpt = CKPT_PATHS[ds]

                    # 本次实验的 sample_steps / p_ratio_lists（传给 build_test_ngds）
                    args.sample_steps = [step]
                    args.p_ratio_lists = [p_ratio]

                    # 其它通用设置
                    args.n_steps = 1
                    args.data_dir = 'data'
                    args.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")

                    log_dir = os.path.join(
                        base_args.save_log,
                        f"diff-{ds}-{model}-step{step}-pr{p_ratio}-{args.timestamp}"
                    )
                    os.makedirs(log_dir, exist_ok=True)
                    args.log_dir = log_dir

                    perf = main(args)

                    # # 把结果写到“不同数据集/模型/参数”的 txt 文件中
                    # result_txt = os.path.join(
                    #     log_dir,
                    #     f"result_{ds}_{model}_step{step}_pr{p_ratio}.txt"
                    # )
                    # with open(result_txt, 'w') as f:
                    #     f.write(f"dataset: {ds}\n")
                    #     f.write(f"model: {model}\n")
                    #     f.write(f"sample_steps: {args.sample_steps}\n")
                    #     f.write(f"p_ratio_lists: {args.p_ratio_lists}\n")
                    #     f.write(f"task_type: {args.task_type}\n")
                    #     f.write("\n")
                    #     for k, v in perf.items():
                    #         f.write(f"{k}: {v}\n")
                    #
                    # logging.info('Finish! Result saved to %s', result_txt)

                    # 关闭并移除当前 handler，避免重复写日志
                    # logging.getLogger().removeHandler(fh)
                    # fh.close()

    logging.info('All experiments finished!')

    # ## n_steps for solver
    # args.n_steps = 1
    # device = torch.device(args.device)
    # if not os.path.exists(args.save_log):
    #     os.makedirs(args.save_log)
    # args.timestamp = datetime.datetime.now().strftime(
    #     "%Y%m%d-%H%M%S-%f")
    # args.data_dir = 'data'
    # log_dir = args.save_log + '/diff-{}-{}-{}'.format(args.dataset, str(args.model),
    #                                                   args.timestamp)
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # args.log_dir = log_dir
    # log_format = '%(asctime)s %(message)s'
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    # fh = logging.FileHandler(os.path.join(log_dir, 'diffusion_aug.log'))
    # fh.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(fh)
    # logging.info('This is the log_dir: {}'.format(log_dir))
    # logging.info('This is the param space: {}'.format(args))
    # main(args)
    #
    # logging.info('Finish! Log_dir final is: {}'.format(log_dir))
