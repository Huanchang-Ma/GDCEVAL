import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
from torch_geometric.data import Data
import numpy as np
from utils_reGen.sde import VESDE
from utils_reGen.solver import LangevinCorrector, ReverseDiffusionPredictor, mask_x, mask_adjs, gen_noise, get_score_fn
from torch_geometric.utils import to_dense_adj
from utils_reGen.mol_utils import convert_dense_adj_to_sparse_with_attr
from evaluation.mmd import guassian_kernel_ma
from utils_reGen.mol_utils import gen_mol
from utils_reGen.graph_utils import quantize_mol
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from utils_reGen.misc import _eval_rocauc
from sklearn.metrics.pairwise import cosine_similarity
import math  # 导入用于判断是否是 NaN
import random
import torch.nn as nn
from visualize_mol import visualize_batch
from visualize_mol import visualize_molecule
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

import torch.nn.functional as F
import logging


def fused_gromov_wasserstein_3d(dist_matrix1, dist_matrix2, features1, features2, alpha=0.5, num_iters=100):
    # Check input dimensions
    assert len(dist_matrix1.shape) == 3, "dist_matrix1 should be a 3D tensor"
    assert len(dist_matrix2.shape) == 3, "dist_matrix2 should be a 3D tensor"
    assert len(features1.shape) == 3, "features1 should be a 3D tensor"
    assert len(features2.shape) == 3, "features2 should be a 3D tensor"
    assert dist_matrix1.shape[0] == dist_matrix2.shape[0], "Batch sizes should match"
    assert features1.shape[0] == features2.shape[0], "Batch sizes should match"

    batch_size = dist_matrix1.shape[0]
    num_nodes1 = dist_matrix1.shape[1]
    num_nodes2 = dist_matrix2.shape[1]

    # Initial uniform coupling for the entire batch
    coupling = torch.ones((batch_size, num_nodes1, num_nodes2)) / (num_nodes1 * num_nodes2)
    coupling = coupling.to(dist_matrix1.device)

    features1 = features1.float()  # 如果需要，可以将 features1 转换为浮点类型
    features2 = features2.float()

    for _ in range(num_iters):
        # Cost matrix for node features
        cost_feature = torch.cdist(features1, features2, p=2)

        # FGW cost matrix
        cost_matrix = alpha * cost_feature + (1 - alpha) * torch.cdist(
            torch.matmul(dist_matrix1, coupling), torch.matmul(dist_matrix2, coupling), p=2
        )

        # Update coupling using Sinkhorn-Knopp algorithm
        coupling = torch.exp(-cost_matrix) * coupling
        # print("coupling", coupling)
        coupling /= coupling.sum(dim=(1, 2), keepdim=True)
        # print("coupling.sum", coupling.sum(dim=(1, 2), keepdim=True))
        # print("coupling", coupling)

    # Compute the final FGW distance
    fgw_distance = (coupling * cost_matrix).sum(dim=(1, 2)).mean()
    # print(fgw_distance.shape, fgw_distance, 'fgw_distance.shape, fgw_distance')
    # assert False
    return fgw_distance

def fused_gromov_wasserstein_ma(dist_matrix1, dist_matrix2, features1, features2, alpha=0.5, num_iters=100):
    if len(dist_matrix1) == 3 & dist_matrix1.size(0) == 1:
        dist_matrix1 = dist_matrix1.squeeze(0)
        dist_matrix2 = dist_matrix2.squeeze(0)
        num_nodes1 = dist_matrix1.size(0)
        num_nodes2 = dist_matrix2.size(0)
        # print(num_nodes1, num_nodes2, 'num_nodes1, num_nodes2')
        # Initial uniform coupling
        coupling = torch.ones((num_nodes1, num_nodes2)) / (num_nodes1 * num_nodes2)
        coupling = coupling.to(dist_matrix1.device)
    else:
        batch_size = dist_matrix1.shape[0]
        num_nodes1 = dist_matrix1.shape[1]
        num_nodes2 = dist_matrix2.shape[1]

        # Concatenate the batch and node dimensions
        dist_matrix1 = dist_matrix1.view(batch_size * num_nodes1, num_nodes1)
        dist_matrix2 = dist_matrix2.view(batch_size * num_nodes2, num_nodes2)
        features1 = features1.view(batch_size * num_nodes1, -1)
        features2 = features2.view(batch_size * num_nodes2, -1)

        # Initial uniform coupling
        coupling = torch.ones((batch_size * num_nodes1, batch_size * num_nodes2)) / (
                batch_size * num_nodes1 * batch_size * num_nodes2)
        coupling = coupling.to(dist_matrix1.device)
        # coupling = torch.ones((num_nodes2, batch_size * num_nodes1)) / (
        #         batch_size * num_nodes1 * num_nodes2)
        # coupling = coupling.to(dist_matrix1.device)

        # 按照 dist_matrix1 和 dist_matrix2 的维度重新调整 coupling 的形状
        new_shape = (dist_matrix1.size(0), dist_matrix2.size(0))
        # 使用 view 方法对 coupling 进行 reshape 操作
        coupling_reshaped = coupling.view(new_shape)
        print(coupling.size())    ## torch.Size([776, 776])
        print(dist_matrix1.size())    ## torch.Size([776, 97])
        print(dist_matrix2.size())    ## torch.Size([776, 97])
    # Iterate to update coupling
    for _ in range(num_iters):
        # Cost matrix for node features
        embeddings = torch.nn.functional.embedding(features2, features1)  # 将节点索引转换为嵌入向量
        cost_feature = torch.cdist(features1, embeddings, p=2)
        # cost_feature = torch.cdist(features1, features2, p=2)
        # cost_matrix = alpha * cost_feature + (1 - alpha) * torch.cdist(dist_matrix1 @ coupling, dist_matrix2 @ coupling, p=2)
        cost_matrix = alpha * cost_feature + (1 - alpha) * torch.cdist(dist_matrix1 @ coupling_reshaped, dist_matrix2 @ coupling_reshaped, p=2)

        # cost_matrix = alpha * torch.sum(dist_matrix1 * coupling, dim=1) + (1 - alpha) * torch.cdist(dist_matrix1 * coupling, dist_matrix2 * coupling, p=2).squeeze(1)        # Update coupling using Sinkhorn-Knopp algorithm
        coupling = torch.exp(-cost_matrix) * coupling
        coupling /= coupling.sum()

    # Compute the final FGW distance
    fgw_distance = (coupling * cost_matrix).sum()
    return fgw_distance

def fused_gromov_wasserstein(dist_matrix1, dist_matrix2, features1, features2, alpha=0.5, num_iters=100):
    if len(dist_matrix1) == 3 & dist_matrix1.size(0) == 1:
        dist_matrix1 = dist_matrix1.squeeze(0)
        dist_matrix2 = dist_matrix2.squeeze(0)
        num_nodes1 = dist_matrix1.size(0)
        num_nodes2 = dist_matrix2.size(0)
        # print(num_nodes1, num_nodes2, 'num_nodes1, num_nodes2')
        # Initial uniform coupling
        coupling = torch.ones((num_nodes1, num_nodes2)) / (num_nodes1 * num_nodes2)
        coupling = coupling.to(dist_matrix1.device)
    else:
        batch_size = dist_matrix1.shape[0]
        num_nodes1 = dist_matrix1.shape[1]
        num_nodes2 = dist_matrix2.shape[1]

        # Concatenate the batch and node dimensions
        dist_matrix1 = dist_matrix1.view(batch_size * num_nodes1, num_nodes1)
        dist_matrix2 = dist_matrix2.view(batch_size * num_nodes2, num_nodes2)
        features1 = features1.view(batch_size * num_nodes1, -1)
        features2 = features2.view(batch_size * num_nodes2, -1)

        # Initial uniform coupling
        coupling = torch.ones((batch_size * num_nodes1, batch_size * num_nodes2)) / (
                batch_size * num_nodes1 * batch_size * num_nodes2)
        coupling = coupling.to(dist_matrix1.device)
    # Iterate to update coupling
    for _ in range(num_iters):
        # Cost matrix for node features
        embeddings = torch.nn.functional.embedding(features2, features1)  # 将节点索引转换为嵌入向量
        cost_feature = torch.cdist(features1, embeddings, p=2)
        # cost_feature = torch.cdist(features1, features2, p=2)
        cost_matrix = alpha * cost_feature + (1 - alpha) * torch.cdist(dist_matrix1 @ coupling, dist_matrix2 @ coupling, p=2)

        # Update coupling using Sinkhorn-Knopp algorithm
        coupling = torch.exp(-cost_matrix) * coupling
        coupling /= coupling.sum()

    # Compute the final FGW distance
    fgw_distance = (coupling * cost_matrix).sum()
    return fgw_distance


def inner_sampling(args, generator, x, adj, sde_x, sde_adj, total_sample_steps, diff_steps, flags=None):
    score_fn_x = get_score_fn(sde_x, generator['model_x'], perturb_ratio=args.perturb_ratio)
    score_fn_adj = get_score_fn(sde_adj, generator['model_adj'], perturb_ratio=args.perturb_ratio)
    snr, scale_eps, n_steps = args.snr, args.scale_eps, args.n_steps
    predictor_obj_x = ReverseDiffusionPredictor('x', sde_x, score_fn_x, False, perturb_ratio=args.perturb_ratio)
    corrector_obj_x = LangevinCorrector('x', sde_x, score_fn_x, snr, scale_eps, n_steps,
                                        perturb_ratio=args.perturb_ratio)
    predictor_obj_adj = ReverseDiffusionPredictor('adj', sde_adj, score_fn_adj, False, perturb_ratio=args.perturb_ratio)
    corrector_obj_adj = LangevinCorrector('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps,
                                          perturb_ratio=args.perturb_ratio)
    x, adj = mask_x(x, flags), mask_adjs(adj, flags)
    total_sample_steps = total_sample_steps
    timesteps = torch.linspace(1, 1e-3, total_sample_steps, device=args.device)[-diff_steps:]
    with torch.no_grad():
        # -------- Reverse diffusion process --------
        for i in range(diff_steps):
            t = timesteps[i]
            vec_t = torch.ones(adj.shape[0], device=t.device) * t
            _x = x
            x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
            adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)
            _x = x
            x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
            adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)
    return x_mean.detach(), adj_mean.detach()


## -------- Main function for augmentation--------##

def ensemble_predictions(y_pred_test_aug, y_pred_test_ori, threshold=0.5):
    """
    Ensemble the predictions from two binary classification models by averaging their results
    and applying a threshold to get the final binary result.

    Args:
    - y_pred_test_aug (torch.Tensor): Predictions from the first model (shape: [64, 1])
    - y_pred_test_ori (torch.Tensor): Predictions from the second model (shape: [64, 1])
    - threshold (float): Threshold to convert averaged predictions to binary results (default: 0.5)

    Returns:
    - torch.Tensor: Final binary predictions after ensembling (shape: [64, 1])
    """
    # Ensure the predictions are in the same shape
    assert y_pred_test_aug.shape == y_pred_test_ori.shape, "Input tensors must have the same shape"

    # Average the predictions
    averaged_predictions = (y_pred_test_aug + y_pred_test_ori) / 2.0

    # Apply the threshold to get binary results
    binary_predictions = (averaged_predictions > threshold).int()

    return binary_predictions


def reverse_adj(adj):
    virtual_edges = 1 - adj[..., :3].sum(dim=-1, keepdims=True)
    adj = torch.cat([adj[..., :3], virtual_edges], dim=-1)

    # Argmax to find the most likely bond type
    adj = adj.argmax(dim=-1)

    # Adjust bond type indices, mapping the last type (3, originally for virtual edges) to 0
    adj = torch.where(adj == 3, torch.zeros_like(adj), adj + 1).float()
    adj_new = adj.clone()
    return adj_new


def reverse_x(x, ptr):
    num_graphs = ptr.size(0) - 1
    node_features_per_graph = ptr[1] - ptr[0]
    reshaped_x = torch.zeros(num_graphs, node_features_per_graph, x.size(1))

    for i in range(num_graphs):
        start_index = ptr[i]
        end_index = ptr[i + 1]
        reshaped_x[i] = x[start_index:end_index]
    x_new = reshaped_x.clone()
    return x_new


# def get_aug_grads(prediction_model, inner_output_data, batch_data, ori_feats, ori_edge, gembs_ori, y_pred_ori,prob_func, criterion, args):
#     prediction_model.eval()
#
#     inner_output_x, inner_output_adj = inner_output_data
#     inner_output_adj = mask_adjs(inner_output_adj, None)
#     inner_output_x = mask_x(inner_output_x, None)
#
#     inner_output_x, inner_output_adj = inner_output_x.requires_grad_(), inner_output_adj.requires_grad_()
#     ori_feats = ori_feats.requires_grad_()
#     ori_edge = ori_edge.requires_grad_()
#
#     with torch.enable_grad():
#
#         inner_x_all, inner_adj_all = inner_output_x, inner_output_adj
#         inner_edge_index, inner_edge_attr = convert_dense_adj_to_sparse_with_attr(inner_adj_all, None)
#         bdata_y = y_pred_ori
#
#         bdata = Data(x=inner_x_all.reshape(batch_data.x.shape).long(), edge_index=inner_edge_index,
#                      edge_attr=inner_edge_attr.float(), y=bdata_y, batch=batch_data.batch)
#
#         if inner_output_x.shape[0] >= 1:
#             inner_output_preds, _, inner_output_gembs = prediction_model(bdata)
#             bdata_target = bdata.y.to(torch.float32)
#
#             if 'classification' in args.task_type:
#                 bdata_probs = prob_func(inner_output_preds.view(bdata.y.size()).to(torch.float32))
#                 loss_y = torch.log(
#                     (bdata_probs * bdata_target + (1 - bdata_probs) * (1 - bdata_target)).clamp_min(
#                         1e-18)).mean()  # maximize log likelihood
#             else:
#                 loss_y = - criterion(inner_output_preds.view(bdata.y.size()).to(torch.float32),
#                                      bdata_target).mean()
#
#             inner_struc_dist = torch.cdist(inner_output_adj, inner_output_adj, p=2)
#             ori_struc_dist = torch.cdist(ori_edge, ori_edge, p=2)
#
#             # Calculate FGW distance between the two graphs
#             loss_fgw_dist = fused_gromov_wasserstein_3d(inner_struc_dist, ori_struc_dist, inner_output_x, ori_feats)
#
#             loss_g_embs = F.smooth_l1_loss(inner_output_gembs, gembs_ori, beta=1.0).mean()
#             loss_strut = - F.mse_loss(inner_output_adj.view(inner_output_x.shape[0], -1),
#                                       ori_edge.view(inner_output_x.shape[0], -1)).mean()
#             total_loss = args.ly_w * loss_y + args.lfgw_w * loss_fgw_dist + args.lgemb_w * loss_g_embs + args.lstrut_w * loss_strut
#             logging.info(
#                 'loss_y = {}, loss_fgw_dist = {}, loss_g_embs = {}, loss_strut = {}'.format(
#                     round(args.ly_w * loss_y.item(), 4),
#                     round(args.lfgw_w * loss_fgw_dist.item(), 4),
#                     round(args.lgemb_w * loss_g_embs.item(), 4),
#                     round(args.lstrut_w * loss_strut.item(), 4)
#                 )
#             )
#             logging.info('Total_loss = {}'.format(round(total_loss.item(), 4)))
#             aug_grad_x, aug_grad_adj = torch.autograd.grad(total_loss, [inner_output_x, inner_output_adj])
#
#         else:
#             aug_grad_x, aug_grad_adj = None, None
#
#     return aug_grad_x, aug_grad_adj, total_loss.item()


# # ==============================================1.===================================================#
# def get_aug_grads(prediction_model, inner_output_data, batch_data, ori_feats, ori_edge, gembs_ori, y_pred_ori,
#                   prob_func, criterion, args):
#     prediction_model.eval()
#
#     inner_output_x, inner_output_adj = inner_output_data
#     inner_output_adj = mask_adjs(inner_output_adj, None)
#     inner_output_x = mask_x(inner_output_x, None)
#
#     inner_output_x = inner_output_x.requires_grad_()
#     inner_output_adj = inner_output_adj.requires_grad_()
#     ori_feats = ori_feats.requires_grad_()
#     ori_edge = ori_edge.requires_grad_()
#
#     with torch.enable_grad():
#         inner_x_all, inner_adj_all = inner_output_x, inner_output_adj
#         inner_edge_index, inner_edge_attr = convert_dense_adj_to_sparse_with_attr(inner_adj_all, None)
#         bdata_y = y_pred_ori
#
#         bdata = Data(
#             x=inner_x_all.reshape(batch_data.x.shape).long(),
#             edge_index=inner_edge_index,
#             edge_attr=inner_edge_attr.float(),
#             y=bdata_y,
#             batch=batch_data.batch
#         )
#
#         if inner_output_x.shape[0] >= 1:
#             inner_output_preds, _, inner_output_gembs = prediction_model(bdata)
#             bdata_target = bdata.y.to(torch.float32)
#
#             if 'classification' in args.task_type:
#                 # 使用标准的二分类损失函数
#                 criterion_cls = nn.BCEWithLogitsLoss()
#                 loss_y = criterion_cls(inner_output_preds.view(-1), bdata_target.view(-1))
#             else:
#                 # 去掉负号，直接最小化损失
#                 loss_y = criterion(
#                     inner_output_preds.view(bdata.y.size()).to(torch.float32),
#                     bdata_target
#                 ).mean()
#
#             inner_struc_dist = torch.cdist(inner_output_adj, inner_output_adj, p=2)
#             ori_struc_dist = torch.cdist(ori_edge, ori_edge, p=2)
#
#             # 计算图之间的FGW距离
#             loss_fgw_dist = fused_gromov_wasserstein_3d(
#                 inner_struc_dist, ori_struc_dist, inner_output_x, ori_feats
#             )
#
#             # 图嵌入的平滑L1损失
#             loss_g_embs = F.smooth_l1_loss(
#                 inner_output_gembs, gembs_ori, beta=1.0
#             ).mean()
#
#             # 去掉负号，直接计算邻接矩阵的MSE损失
#             loss_strut = F.mse_loss(
#                 inner_output_adj.view(inner_output_adj.size(0), -1),
#                 ori_edge.view(ori_edge.size(0), -1)
#             ).mean()
#
#             # 总损失
#             total_loss = (
#                     args.ly_w * loss_y +
#                     args.lfgw_w * loss_fgw_dist +
#                     args.lgemb_w * loss_g_embs +
#                     args.lstrut_w * loss_strut
#             )
#
#             logging.info(
#                 'loss_y = {}, loss_fgw_dist = {}, loss_g_embs = {}, loss_strut = {}'.format(
#                     round(args.ly_w * loss_y.item(), 4),
#                     round(args.lfgw_w * loss_fgw_dist.item(), 4),
#                     round(args.lgemb_w * loss_g_embs.item(), 4),
#                     round(args.lstrut_w * loss_strut.item(), 4)
#                 )
#             )
#             logging.info('Total_loss = {}'.format(round(total_loss.item(), 4)))
#
#             aug_grad_x, aug_grad_adj = torch.autograd.grad(
#                 total_loss, [inner_output_x, inner_output_adj]
#             )
#         else:
#             aug_grad_x, aug_grad_adj = None, None
#
#     return aug_grad_x, aug_grad_adj, total_loss.item()

# # ==============================================2.===================================================#
def get_aug_grads(prediction_model, inner_output_data, batch_data, ori_feats, ori_edge, gembs_ori, y_pred_ori,
                  prob_func, criterion, args):
    prediction_model.eval()

    inner_output_x, inner_output_adj = inner_output_data
    inner_output_adj = mask_adjs(inner_output_adj, None)
    inner_output_x = mask_x(inner_output_x, None)

    inner_output_x = inner_output_x.requires_grad_()
    inner_output_adj = inner_output_adj.requires_grad_()
    ori_feats = ori_feats.requires_grad_()
    ori_edge = ori_edge.requires_grad_()

    with torch.enable_grad():

        inner_x_all, inner_adj_all = inner_output_x, inner_output_adj
        inner_edge_index, inner_edge_attr = convert_dense_adj_to_sparse_with_attr(inner_adj_all, None)
        bdata_y = y_pred_ori

        repeated_inner_output_x = inner_x_all.reshape(batch_data.x.shape).float()

        bdata = Data(x=repeated_inner_output_x, edge_index=inner_edge_index, edge_attr=inner_edge_attr.float(),
                     y=bdata_y, batch=batch_data.batch)

        if inner_output_x.shape[0] >= 1:
            inner_output_preds, _, inner_output_gembs = prediction_model(bdata)
            bdata_target = bdata.y.to(torch.float32)

            if 'classification' in args.task_type:
                # 使用标准的二分类损失函数
                criterion_cls = nn.BCEWithLogitsLoss()
                loss_y = criterion_cls(inner_output_preds.view(-1), bdata_target.view(-1))
            else:
                # 去掉负号，直接最小化损失
                loss_y = - criterion(inner_output_preds.view(bdata.y.size()).to(torch.float32), bdata_target).mean()

            # 改进部分开始

            # 1. 计算图之间的 FGW 距离 loss_fgw_dist
            # 计算结构距离矩阵
            inner_struc_dist = torch.cdist(inner_output_adj, inner_output_adj, p=2) ** 2 + 1e-8
            ori_struc_dist = torch.cdist(ori_edge, ori_edge, p=2) ** 2 + 1e-8

            # 计算特征距离矩阵
            inner_feat_dist = torch.cdist(inner_output_x, inner_output_x, p=2) ** 2
            ori_feat_dist = torch.cdist(ori_feats, ori_feats, p=2) ** 2

            # 计算 FGW 距离
            loss_fgw_dist = fused_gromov_wasserstein_3d(inner_struc_dist, ori_struc_dist, inner_feat_dist,
                                                        ori_feat_dist)
            print(loss_fgw_dist)

            # 2. 计算节点级别的权重并聚合到图级别
            node_weights = repeated_inner_output_x.norm(p=2, dim=1)  # [num_nodes]
            batch_size = int(bdata.batch.max().item()) + 1
            device = repeated_inner_output_x.device
            graph_weights = torch.zeros(batch_size, device=device)  # [batch_size]
            graph_weights = graph_weights.index_add(0, bdata.batch, node_weights)  # 聚合到对应的图上
            graph_weights = graph_weights / graph_weights.sum()

            # 3. 计算图级别的嵌入损失
            loss_g_embs = F.smooth_l1_loss(inner_output_gembs, gembs_ori, beta=1.0, reduction='none')
            loss_g_embs = loss_g_embs.mean(dim=1)  # [batch_size]

            # 4. 计算加权的图嵌入损失
            loss_g_embs = (loss_g_embs * graph_weights).sum()

            # 5. 计算结构损失，去掉负号，直接最小化 MSE 损失
            loss_strut = F.mse_loss(inner_output_adj.view(inner_output_adj.size(0), -1),
                                    ori_edge.view(ori_edge.size(0), -1)).mean()

            # 总损失
            total_loss = (args.ly_w * loss_y +
                          args.lfgw_w * loss_fgw_dist +
                          args.lgemb_w * loss_g_embs +
                          args.lstrut_w * loss_strut)

            # 改进部分结束

            logging.info(
                'loss_y = {}, loss_fgw_dist = {}, loss_g_embs = {}, loss_strut = {}'.format(
                    round(args.ly_w * loss_y.item(), 4),
                    round(args.lfgw_w * loss_fgw_dist.item(), 4),
                    round(args.lgemb_w * loss_g_embs.item(), 4),
                    round(args.lstrut_w * loss_strut.item(), 4)
                )
            )
            logging.info('Total_loss = {}'.format(round(total_loss.item(), 4)))
            aug_grad_x, aug_grad_adj = torch.autograd.grad(total_loss, [inner_output_x, inner_output_adj])

        else:
            aug_grad_x, aug_grad_adj = None, None

    return aug_grad_x, aug_grad_adj, total_loss.item()


def build_test_ngds(args, testloader, model, generator):
    if 'classification' in args.task_type:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        prob_func = torch.nn.Sigmoid()
    else:
        criterion = torch.nn.MSELoss(reduction='none')
        prob_func = None

    y_pred_test_aug_ls = []
    y_ground_true_ls = []
    y_pred_test_ori_ls = []
    y_pred_test_aug_avg_ls = []
    mmd_value_set = []
    y_pred_test_aug_ls_binary2 = []
    y_pred_test_ori_ls_binary2 = []
    y_pred_test_aug_ls_binary1 = []
    y_pred_test_ori_ls_binary1 = []

    mmd_value = 0
    count = 0
    # max_mmd = 0
    # min_mmd = 1000
    # print(args.sample_steps) ##20
    sample_steps_value = 5  ## 将args.sample_steps中列表[20]的数值提取出来
    validity_correction = 0
    validity_correction_sum = 0
    test_data_add = 0
    threshold_samples = 0.5

    for step, te_batch_data in enumerate(testloader):
        # print("te_batch_data.y:", te_batch_data.y)    ## 0 1
        # print("te_batch_data.y.detach().cpu().numpy():", te_batch_data.y.detach().cpu().numpy())
        # with open('output_y.txt', 'a') as file:
        #     # file_samples.write(f"--------------------------------------------------------------\n")
        #     file.write(f"te_batch_data.y:{te_batch_data.y}\n")
        #     file.write(f"te_batch_data.y.detach().cpu().numpy():{te_batch_data.y.detach().cpu().numpy()}\n\n")    ##
        #     file.write(f"--------------------------------------------------------------\n")
        # file.close()

        model.eval()
        te_batch_data = te_batch_data.to(args.device)
        # print("te_batch_data.y:", te_batch_data.y)    ## 0 1
        # print(te_batch_data.x)    ## 0-1
        # print("te_batch_data.edge_index", te_batch_data.edge_index)
        # print("te_batch_data.edge_attr", te_batch_data.edge_attr)
        ori_feats = reverse_x(te_batch_data.x, te_batch_data.ptr).to(args.device)  ## 处理原始的特征和邻接矩阵(用于SDE(随机扩散过程)的格式)
        ori_edge = reverse_adj(
            to_dense_adj(te_batch_data.edge_index, batch=te_batch_data.batch, edge_attr=te_batch_data.edge_attr)).to(
            args.device)
        ori_adj = to_dense_adj(te_batch_data.edge_index, batch=te_batch_data.batch, edge_attr=te_batch_data.edge_attr)
        # print(ori_feats, ori_edge)    ## int

        with (torch.no_grad()):
            te_y_pred, _, te_gembs = model(te_batch_data)  ## te_y_pred 是初始预测测试集的标签
            # sde
            total_loss_set = []
            aug_new_data_set = []
            # print("经过model(te_batch_data)的te_y_pred:", te_y_pred)    ## 经过model(te_batch_data)的te_y_pred: tensor([[-0.1721], [ 2.8542], [ 2.4656], [ 4.1752],
            # te_y_pred_binary1 = torch.argmax(te_y_pred, dim=1)    ## tensor([0, 0, 0, 0, 0......
            te_y_pred_binary1 = F.softmax(te_y_pred, dim=1).detach().cpu()
            te_y_pred_prob = torch.exp(te_y_pred)
            # 归一化概率分布使其和为1
            te_y_pred_prob /= torch.sum(te_y_pred_prob, dim=1, keepdim=True)
            # 使用argmax获取预测的类别
            te_y_pred_binary2 = torch.round(te_y_pred_prob)
            # print("te_y_pred_binary1 te_y_pred_binary2分别为:", te_y_pred_binary1, te_y_pred_binary2)    ## te_y_pred_binary1 te_y_pred_binary2二者输出一致

            if len(args.p_ratio_lists) > 1 & len(args.sample_steps) > 1:
                logging.info('Error!, Can not deal with double index!')
                assert False

            print(args.p_ratio_lists)  ## [0.1, 0.3, 0.5, 0.7, 0.9]    取 0.1
            print("len(args.p_ratio_lists:", len(args.p_ratio_lists))
            # print(args.sample_steps)

            for p_ratio in args.p_ratio_lists:
                args.perturb_ratio = p_ratio  ## perturb_ratio 和 sample_steps用于控制随机扩散过程的参数。
                mmd_value_sum = 0
                count_mmd = 0
                max_mmd = 0
                min_mmd = 1000
                max_index = 0
                max_validity_correction = 0
                min_validity_correction = 1000
                sum_validity_correction = 0
                # for sample_steps in args.sample_steps:   ## print(args.sample_steps)的值为[20],当前sample_steps循环只遍历1次且每一次Forward steps = 20
                # sample_steps_value = args.sample_steps[0]
                mmd_samples_value_set = []
                fgw_distance_set = []
                fgw_distance_set_3d = []
                similarity_matrix_set = []
                val_cor_value_set = []
                y_pred_test_aug_ls_samples = []
                y_pred_test_aug_avg_ls_samples = []
                y_pred_ori_np_samples = []
                y_pred_test_aug_ls_samples_binary1 = []    ## softmax
                y_pred_test_ori_ls_samples_binary1 = []    ## softmax
                y_pred_test_aug_ls_samples_binary2 = []
                y_pred_test_ori_ls_samples_binary2 = []
                for sample_steps in range(1, sample_steps_value + 1):
                    print('Forward steps =', sample_steps, 'with p_ratio =', args.perturb_ratio)
                    # assert False
                    total_sample_steps = sample_steps    ## VESDE随total_sample_steps变化
                    # total_sample_steps = 20
                    # total_sample_steps = 30
                    sde_x = VESDE(sigma_min=0.1, sigma_max=1, N=total_sample_steps)  ## 随机扩散过程
                    sde_adj = VESDE(sigma_min=0.1, sigma_max=1, N=total_sample_steps)

                    # perturb x
                    peturb_t = torch.ones(ori_edge.shape[0]).to(args.device) * (sde_adj.T - 1e-3) + 1e-3
                    mean_x, std_x = sde_x.marginal_prob(ori_feats, peturb_t)
                    z_x = gen_noise(ori_feats, flags=None, sym=False,  ## gen_noise 函数用于生成噪声数据
                                    perturb_ratio=args.perturb_ratio)
                    perturbed_x = mean_x + std_x[:, None, None] * z_x
                    perturbed_x = mask_x(perturbed_x, None)  ## mask_x 和 mask_adjs 函数用于处理数据掩码

                    # perturb adj
                    mean_adj, std_adj = sde_adj.marginal_prob(ori_edge, peturb_t)
                    z_adj = gen_noise(ori_edge, flags=None, sym=True,
                                      perturb_ratio=args.perturb_ratio)
                    perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
                    perturbed_adj = mask_adjs(perturbed_adj, None)

                    timesteps = torch.linspace(1, 1e-3, total_sample_steps, device=args.device)[-sample_steps:]
                    # logging.info('timesteps_in = {}'.format(timesteps))

                    score_fn_x = get_score_fn(sde_x, generator['model_x'], perturb_ratio=args.perturb_ratio)
                    score_fn_adj = get_score_fn(sde_adj, generator['model_adj'], perturb_ratio=args.perturb_ratio)

                    # ## 判断score是否为0    score值不为0
                    # if score_fn_x == 0:
                    #     print("!!!!!!Score for x is 0!!!!!!")
                    # if score_fn_adj == 0:
                    #     print("!!!!!!Score for adj is 0!!!!!!")

                    predictor_obj_x = ReverseDiffusionPredictor('x', sde_x, score_fn_x, False,
                                                                ## ReverseDiffusionPredictor 和 LangevinCorrector 类用于预测和校正潜在变量
                                                                perturb_ratio=args.perturb_ratio)
                    corrector_obj_x = LangevinCorrector('x', sde_x, score_fn_x, args.snr, args.scale_eps, args.n_steps,
                                                        perturb_ratio=args.perturb_ratio)
                    predictor_obj_adj = ReverseDiffusionPredictor('adj', sde_adj, score_fn_adj, False,
                                                                  perturb_ratio=args.perturb_ratio)
                    corrector_obj_adj = LangevinCorrector('adj', sde_adj, score_fn_adj, args.snr, args.scale_eps,
                                                          args.n_steps,
                                                          perturb_ratio=args.perturb_ratio)
                    outer_iters = range(sample_steps)
                    print("outer_iters:", outer_iters)


                    for i in outer_iters:
                        inner_output_x, inner_output_adj = inner_sampling(args, generator, perturbed_x, perturbed_adj,
                                                                          ## 实际采样过程，为了生成新的数据样本
                                                                          sde_x,
                                                                          sde_adj, sample_steps, sample_steps - i,
                                                                          None)

                        aug_grad_x, aug_grad_adj, total_loss = get_aug_grads(model, [inner_output_x, inner_output_adj],
                                                                             ## get_aug_grads 函数计算了对抗性增强的梯度
                                                                             te_batch_data,
                                                                             ori_feats, ori_edge,
                                                                             te_gembs, te_y_pred, prob_func, criterion,
                                                                             args)
                        with torch.no_grad():
                            t = timesteps[i]
                            vec_t = torch.ones(perturbed_adj.shape[0], device=t.device) * t
                            _x = perturbed_x
                            perturbed_x, perturbed_x_mean = corrector_obj_x.update_fn(perturbed_x, perturbed_adj,
                                                                                      ## corrector_obj_x.update_fn 和 predictor_obj_x.update_fn 函数用于更新潜在变量
                                                                                      None, vec_t,
                                                                                      aug_grad=aug_grad_x)
                            perturbed_adj, perturbed_adj_mean = corrector_obj_adj.update_fn(_x, perturbed_adj,
                                                                                            None,
                                                                                            vec_t,
                                                                                            aug_grad=aug_grad_adj)
                            _x = perturbed_x
                            perturbed_x, perturbed_x_mean = predictor_obj_x.update_fn(perturbed_x, perturbed_adj,
                                                                                      None, vec_t,
                                                                                      aug_grad=aug_grad_x)
                            perturbed_adj, perturbed_adj_mean = predictor_obj_adj.update_fn(_x, perturbed_adj,
                                                                                            None,
                                                                                            vec_t,
                                                                                            aug_grad=aug_grad_adj)

                    # note that only save the last total loss  ## 只保留最后一次的 total loss
                    total_loss_set.append(total_loss)  ## 在循环for sample_steps in range(1, sample_steps_value+1):
                    test_data_add += 1

                    # print(perturbed_x.shape, perturbed_adj.shape)   ## perturbed_x.shape, perturbed_adj.shape分别为 torch.Size([64, 97, 8]) torch.Size([64, 97, 97])
                    # print(perturbed_x_mean.shape, perturbed_adj_mean.shape)    ## perturbed_x_mean.shape, perturbed_adj_mean.shape分别为torch.Size([64, 97, 8])torch.size([64, 97, 97])
                    # perturbed_first_xrow = perturbed_x[0]
                    # print("perturbed_x第一行是：", perturbed_first_xrow)
                    # perturbed_first_adjrow = perturbed_adj[0]
                    # print("perturbed_adj第一行是：", perturbed_first_adjrow)
                    # perturbed_mean_first_xrow = perturbed_x_mean[0]
                    # print("perturbed_x_mean第一行是：", perturbed_mean_first_xrow)
                    # perturbed_mean_first_adjrow = perturbed_adj_mean[0]
                    # print("perturbed_adj_mean第一行是：", perturbed_mean_first_adjrow)

                    ########### validity w/o correction ############
                    perturbed_x_mean_validity = perturbed_x_mean  ## max:0.90625  min:0
                    perturbed_adj_mean_validity = perturbed_adj_mean
                    # print(perturbed_adj_mean_validity)
                    # assert False
                    # perturbed_x_mean_validity = mask_x(perturbed_x_mean_validity, None)
                    # perturbed_adj_mean_validity = mask_adjs(perturbed_adj_mean_validity, None)
                    # perturbed_x_mean_validity = perturbed_x    ## max:0.921875  min:0
                    # perturbed_adj_mean_validity = perturbed_adj
                    ################################################

                    perturbed_adj_mean = mask_adjs(perturbed_adj_mean, None)  ## 0, 1
                    perturbed_x_mean = mask_x(perturbed_x_mean, None)
                    # consider whether reorganize x to 0,1
                    # perturbed_x_mean = (perturbed_x_mean > 0).int()
                    perturbed_x_mean = torch.where(perturbed_x_mean > 0.5, 1, 0)
                    augment_edge_index, augment_edge_attr = convert_dense_adj_to_sparse_with_attr(perturbed_adj_mean,
                                                                                                  None)
                    # print(perturbed_x_mean)
                    aug_data_y = te_y_pred
                    aug_new_data = Data(x=perturbed_x_mean.reshape(te_batch_data.x.shape),
                                        edge_index=augment_edge_index,
                                        edge_attr=augment_edge_attr.float(), y=aug_data_y,
                                        batch=te_batch_data.batch)
                    # 检查原子特征
                    ##Atom features: torch.Size([8, 97, 8])  aug_new_data Atom features: torch.Size([776, 8])
                    # print("Atom features:", perturbed_x_mean.shape)  #Atom features: torch.Size([8, 97, 8])
                    # print("aug_new_data Atom features:", aug_new_data.x.shape)  # aug_new_data Atom features: torch.Size([776, 8])
                    # print("aug_new_data batch:", aug_new_data.batch)
                    # print(aug_new_data.edge_index)
                    # print(aug_new_data.edge_attr)
                    # print(aug_new_data.batch)
                    # print("augment_edge_index:", augment_edge_index)
                    # print("augment_edge_attr:", augment_edge_attr)
                    # assert False
                    # atom_counts = calculate_atom_counts(aug_new_data)
                    # visualize_batch(aug_new_data, aug_new_data.edge_index, aug_new_data.edge_attr, atom_counts)
                    # assert False
                    # visualize_batch(aug_new_data, output_dir='./mol_vis')

                    # print("Sample atom feature vector:", perturbed_x_mean[0])
                    # 检查边特征
                    # print("Edge features:", augment_edge_attr)
                    # print("Sample edge feature vector:", augment_edge_attr[0])
                    # print(f"aug_new_data.edge_index shape: {aug_new_data.edge_index.shape}")
                    # print(f"aug_new_data.edge_attr shape: {aug_new_data.edge_attr.shape}")
                    # aug_new_data.edge_index shape: torch.Size([2, 389])
                    # aug_new_data.edge_attr shape: torch.Size([389, 3])
                    # print(aug_new_data.edge_index)
                    # print(aug_new_data.batch)
                    # assert False

                    # print(aug_new_data.x)
                    aug_new_data_adj = reverse_adj(to_dense_adj(aug_new_data.edge_index, batch=aug_new_data.batch, edge_attr=aug_new_data.edge_attr)).to(args.device)
                    # print(aug_new_data_adj)
                    aug_new_data_set.append(aug_new_data)
                    # assert False

                    #############################  添加数据选择策略validity w/o correction  ###############################
                    # 添加数据选择策略validity w/o correction(全部)
                    ## self.configt.data.data: ogbg-molbace

                    samples_int = quantize_mol(perturbed_adj_mean_validity)
                    samples_int = samples_int - 1
                    samples_int[samples_int == -1] = 3  # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2
                    # print("samples_int", samples_int)

                    samples_int[samples_int > 3] = 3
                    perturbed_adj_mean_validity = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3, 1, 2)
                    perturbed_x_mean_validity = torch.where(perturbed_x_mean_validity > 0.5, 1, 0)
                    perturbed_x_mean_validity = torch.concat([perturbed_x_mean_validity, 1 - perturbed_x_mean_validity.sum(dim=-1, keepdim=True)], dim=-1)  # 32, 9, 4 -> 32, 9, 5

                    # first_xrow = perturbed_x_mean_validity[0]
                    # print("perturbed_x_mean_validity第一行是：", first_xrow)
                    # first_adjrow = perturbed_adj_mean_validity[0]
                    # print("perturbed_adj_mean_validity第一行是：", first_adjrow)
                    print(args.dataset)  ## ogbg-molbace
                    gen_mols, num_mols_wo_correction = gen_mol(perturbed_x_mean_validity, perturbed_adj_mean_validity,
                                                               args.dataset)
                    num_mols = len(gen_mols)
                    # print(gen_mols)
                    print("num_mols_wo_correction is ", num_mols_wo_correction)
                    # print("num_mols is ", num_mols)    ## 64
                    if num_mols != 0:
                        validity_correction = num_mols_wo_correction / num_mols
                        print("validity w/o correction:", validity_correction)
                    else:
                        validity_correction = -1
                        print("num_mols is zero.")

                    val_cor_value_set.append(validity_correction)

                    if validity_correction > max_validity_correction:
                        max_validity_correction = validity_correction
                        max_index = len(aug_new_data_set) - 1  ## current_index
                    if validity_correction < min_validity_correction:
                        min_validity_correction = validity_correction
                    print("max validity w/o correction:", max_validity_correction)
                    sum_validity_correction += validity_correction

                    ## 计算每一轮sample_steps下aug_new_data与test_data的MMD
                    te_kernel_matrix1 = guassian_kernel_ma(te_batch_data.x, te_batch_data.x)
                    aug_kernel_matrix1 = guassian_kernel_ma(aug_new_data.x, aug_new_data.x)
                    te_batch_kernel_mean1 = torch.mean(te_kernel_matrix1)
                    aug_batch_kernel_mean1 = torch.mean(aug_kernel_matrix1)
                    mmd_samples_value = (te_batch_kernel_mean1 + aug_batch_kernel_mean1 - 2 * torch.mean(
                        guassian_kernel_ma(te_batch_data.x, aug_new_data.x)))
                    mmd_samples_value_set.append(mmd_samples_value)

                    ## FGW Distance
                    # fgw_distance = fused_gromov_wasserstein(ori_edge, aug_new_data_adj, ori_feats, aug_new_data.x, alpha=0.5, num_iters=100)
                    # fgw_distance_3d = fused_gromov_wasserstein_3d(ori_edge, aug_new_data_adj, ori_feats, aug_new_data.x, alpha=0.5, num_iters=100)
                    # print("ori_edge shape:", ori_edge.shape, "type:", type(ori_edge))
                    # print("aug_new_data_adj shape:", aug_new_data_adj.shape, "type:", type(aug_new_data_adj))
                    # print("ori_feats shape:", ori_feats.shape, "type:", type(ori_feats))
                    # print("perturbed_x_mean shape:", perturbed_x_mean.shape, "type:", type(perturbed_x_mean))
                    # print(perturbed_x_mean)
                    ## ori_edge shape: torch.Size([8, 97, 97]) type: <class 'torch.Tensor'>    aug_new_data_adj shape: torch.Size([8, 97, 97]) type: <class 'torch.Tensor'>  ori_feats shape: torch.Size([8, 97, 8]) type: <class 'torch.Tensor'>  aug_new_data.x shape: torch.Size([776, 8]) type: <class 'torch.Tensor'>
                    # fgw_distance = fused_gromov_wasserstein_ma(ori_edge, aug_new_data_adj, ori_feats, perturbed_x_mean, alpha=0.5, num_iters=100)
                    # fgw_distance_set.append(fgw_distance)
                    # print(ori_edge)
                    # print(aug_new_data_adj)
                    # assert False
                    fgw_distance_3d = fused_gromov_wasserstein_3d(ori_edge, aug_new_data_adj, ori_feats, perturbed_x_mean,
                                                               alpha=0.5, num_iters=100)
                    fgw_distance_set_3d.append(fgw_distance_3d)
                    # fgw_distance_set_3d.append(fgw_distance_3d)

                    ## similarity_matrix余弦相似度
                    # 计算图的特征向量（将邻接矩阵展平）
                    ori_edge_flat = ori_edge.flatten().reshape(1, -1)
                    aug_new_data_adj_flat = aug_new_data_adj.flatten().reshape(1, -1)
                    # 将 GPU tensor 移动到 CPU
                    ori_edge_flat_cpu = ori_edge_flat.cpu().numpy()  # 转换为 NumPy 数组
                    aug_new_data_adj_flat_cpu = aug_new_data_adj_flat.cpu().numpy()  # 转换为 NumPy 数组
                    # 计算余弦相似度
                    similarity_matrix = cosine_similarity(ori_edge_flat_cpu, aug_new_data_adj_flat_cpu)
                    # print(similarity_matrix)
                    # assert False
                    similarity_matrix_set.append(similarity_matrix)

                    ## 计算每一轮sample_steps下y_pred_test_aug_samples
                    y_pred_test_aug_samples_, _, te_aug_gembs_samples = model(aug_new_data)  ## 使用model来生成预测(增强数据后的预测结果)
                    if args.task_type == 'regression':
                        y_pred_test_aug_samples = y_pred_test_aug_samples_.detach().cpu()
                        y_pred_test_ori_samples = te_y_pred.detach().cpu()
                        average_gemb_samples = args.te_aug_ratio * te_aug_gembs_samples + (1 - args.te_aug_ratio) * te_gembs
                        y_pred_test_aug_avg_samples = model.predictor(average_gemb_samples).detach().cpu()
                        # print(args.label_names)
                        # y_ground_true_samples = te_batch_data.y[:, args.label_names.index(args.label)].reshape(-1, 1).detach().cpu().reshape(-1).numpy()
                        y_ground_true_samples = te_batch_data.y.detach().cpu()
                        y_pred_test_aug_avg_ls_samples.append(y_pred_test_aug_avg_samples)
                    elif args.task_type == 'classification':
                        y_pred_test_aug_samples = y_pred_test_aug_samples_.detach().cpu()  ## y_pred_test_aug_ 是增强数据后的预测结果
                        # print("y_pred_test_aug_samples_:", y_pred_test_aug_samples_)    ## tensor([[-9.7513e+02],......
                        # print("y_pred_test_aug_samples:", y_pred_test_aug_samples)    ## tensor([[-9.7513e+02],......
                        y_pred_test_ori_samples = te_y_pred.detach().cpu().numpy()  ## test_ori
                        # print("te_batch_data.y:", te_batch_data.y)    ## 0 1
                        ensemble_result_samples = ensemble_predictions(y_pred_test_aug_samples,
                                                                       y_pred_test_ori_samples)  ## ensemble_predictions 函数用于组合不同样本的预测结果
                        y_ground_true_samples = te_batch_data.y.detach().cpu().numpy()  ## ground_true
                        ## 0 1
                        y_pred_test_ori_samples_prob = torch.exp(te_y_pred)
                        # 归一化概率分布使其和为1
                        y_pred_test_ori_samples_prob /= torch.sum(y_pred_test_ori_samples_prob, dim=1, keepdim=True)
                        # 使用argmax获取预测的类别
                        y_pred_test_ori_samples__ = torch.round(y_pred_test_ori_samples_prob)
                        y_pred_test_ori_samples_binary2 = y_pred_test_ori_samples__.detach().cpu().numpy()

                        y_pred_test_ori_samples_binary1 = F.softmax(te_y_pred, dim=1).detach().cpu().numpy()
                        y_pred_test_aug_samples_binary1 = F.softmax(y_pred_test_aug_samples_, dim=1).detach().cpu()

                        y_pred_test_aug_samples_prob = torch.exp(y_pred_test_aug_samples_)
                        # 归一化概率分布使其和为1
                        y_pred_test_aug_samples_prob /= torch.sum(y_pred_test_aug_samples_prob, dim=1, keepdim=True)
                        # 使用argmax获取预测的类别
                        y_pred_test_aug_samples__ = torch.round(y_pred_test_aug_samples_prob)
                        y_pred_test_aug_samples_binary2 = y_pred_test_aug_samples__.detach().cpu()
                        y_pred_test_aug_avg_ls_samples.append(ensemble_result_samples)
                    y_pred_test_aug_ls_samples.append(y_pred_test_aug_samples)

                    if args.task_type == 'classification':
                        y_pred_test_ori_ls_samples_binary1.append(y_pred_test_ori_samples_binary1)  ## F.softmax
                        y_pred_test_aug_ls_samples_binary1.append(y_pred_test_aug_samples_binary1)  ## F.softmax

                        y_pred_test_aug_ls_samples_binary2.append(y_pred_test_aug_samples_binary2)  ## y_pred_test_aug_samples
                        y_pred_test_ori_ls_samples_binary2.append(y_pred_test_ori_samples_binary2)
                    # y_pred_test_ori_ls_samples_binary1.append(y_pred_test_ori_samples_binary1)  ## F.softmax
                    # y_pred_test_aug_ls_samples_binary1.append(y_pred_test_aug_samples_binary1)  ## F.softmax
                    #
                    # y_pred_test_aug_ls_samples_binary2.append(y_pred_test_aug_samples_binary2)  ## y_pred_test_aug_samples
                    # y_pred_test_ori_ls_samples_binary2.append(y_pred_test_ori_samples_binary2)

                    # print("y_pred_test_aug_ls_samples_binary1:", y_pred_test_aug_ls_samples_binary1)
                    # print("\ny_pred_test_aug_ls_samples_binary2:", y_pred_test_aug_ls_samples_binary2)
                    # ## 判断 y_pred_test_aug_ls_samples_binary1与y_pred_test_aug_ls_samples_binary2是否一致    ## 结果显示二者一致
                    # are_equal = torch.equal(y_pred_test_aug_samples_binary1, y_pred_test_aug_samples_binary2)
                    # if are_equal == True:
                    #     with open('output.txt', 'a') as file:
                    #         file.write(f"二者一致\n")
                    #     file.close()
                    # else:
                    #     with open('output.txt', 'a') as file:
                    #         file.write(f"不一致！！\n")
                    #     file.close()
                    # ## 查看 y_pred_test_aug_samples_binary1, y_pred_test_aug_samples_binary2 是否均存在nan值
                    # with open('output_y.txt', 'a') as file:    ## output_y09.. .txt
                    #     file.write(f"--------------------------------------------------------------\n")
                    #     file.write(f"y_ground_true_samples：{y_ground_true_samples}\n")
                    #     file.write(f"y_pred_test_aug_samples_binary：{y_pred_test_aug_samples_binary1}\n")
                    #     file.write(f"y_pred_test_ori_samples_binary：{y_pred_test_ori_samples_binary2}\n\n")
                    # file.close()

                ##计算全部sample_steps下的y
                y_pred_ori_np_samples = np.repeat(y_pred_test_ori_samples, sample_steps, axis=0)  ## test_ori
                y_true_np_samples = np.repeat(y_ground_true_samples, sample_steps, axis=0)  ## ground_true
                y_pred_aug_np_samples = torch.cat(y_pred_test_aug_ls_samples, dim=0).numpy()

                y_pred_ori_np_samples_binary = np.repeat(y_pred_test_ori_samples_binary1, sample_steps,
                                                         axis=0)  ## 为了计算acc
                # y_pred_aug_np_samples_binary = torch.cat(y_pred_test_aug_ls_samples_binary2, dim=0).numpy()  ## 为了计算acc
                y_pred_aug_np_samples_binary = torch.cat(y_pred_test_aug_ls_samples_binary1, dim=0).numpy()  ## 为了计算acc

                # print("y_true_np_samples", y_true_np_samples)
                # print("\ny_pred_ori_np_samples_binary", y_pred_ori_np_samples_binary)
                # print("\ny_pred_aug_np_samples_binary", y_pred_aug_np_samples_binary)
                # with open('output_y.txt', 'a') as file:    ## output_y09.. .txt
                #     # file_samples.write(f"--------------------------------------------------------------\n")
                #     file.write(f"y_true_np_samples：{y_true_np_samples}\n")
                #     file.write(f"y_pred_ori_np_samples_binary：{y_pred_ori_np_samples_binary}\n\n")
                #     file.write(f"y_pred_aug_np_samples_binary：{y_pred_aug_np_samples_binary}\n\n")
                # file.close()

                ############################################### STRATEGY ###############################################
                ## Average validity_correction
                avg_validity_correction = sum_validity_correction / sample_steps
                max_min_validity_correction = max_validity_correction - min_validity_correction
                ## Average mmd && Average fgw
                avg_mmd_samples_value = sum(mmd_samples_value_set) / len(mmd_samples_value_set)
                max_mmd_samples_value = max(mmd_samples_value_set)
                min_mmd_samples_value = min(mmd_samples_value_set)
                max_min_mmd_samples_value = max(mmd_samples_value_set)-min(mmd_samples_value_set)
                # avg_fgw_distance_value = sum(fgw_distance_set) / len(fgw_distance_set)
                # max_fgw_distance_value = max(fgw_distance_set)
                # min_fgw_distance_value = min(fgw_distance_set)
                # max_min_fgw_distance_value = max(fgw_distance_set)-min(fgw_distance_set)
                avg_fgw_distance_value = sum(fgw_distance_set_3d) / len(fgw_distance_set_3d)
                max_fgw_distance_value = max(fgw_distance_set_3d)
                min_fgw_distance_value = min(fgw_distance_set_3d)
                max_min_fgw_distance_value = max(fgw_distance_set_3d) - min(fgw_distance_set_3d)
                ## Medium
                fgw_distance_list = list(fgw_distance_set_3d)
                mmd_samples_value_list = list(mmd_samples_value_set)
                val_cor_list = list(val_cor_value_set)
                total_loss_list = list(total_loss_set)
                similarity_matrix_list = list(similarity_matrix_set)
                # 对fgw_distance_list列表进行排序
                sorted_fgw_distance_list = sorted(fgw_distance_list)
                n = len(fgw_distance_list)
                median_fgw_value = sorted_fgw_distance_list[n // 2] if n % 2 != 0 else (sorted_fgw_distance_list[n // 2 - 1] + sorted_fgw_distance_list[n // 2]) / 2
                median_fgw_index = fgw_distance_list.index(sorted_fgw_distance_list[n // 2])
                median_fgw_mmd_value = mmd_samples_value_list[median_fgw_index]
                median_fgw_val_cor_value = val_cor_list[median_fgw_index]
                median_fgw_loss_value = total_loss_list[median_fgw_index]
                # 对mmd_samples_value_list列表进行排序
                sorted_mmd_samples_value_list = sorted(mmd_samples_value_list)
                n = len(mmd_samples_value_list)
                median_mmd_value = sorted_mmd_samples_value_list[n // 2] if n % 2 != 0 else (sorted_mmd_samples_value_list[n // 2 - 1] + sorted_mmd_samples_value_list[n // 2]) / 2
                median_mmd_index = mmd_samples_value_list.index(sorted_mmd_samples_value_list[n // 2])
                median_mmd_fgw_value = fgw_distance_list[median_mmd_index]
                median_mmd_val_cor_value = val_cor_list[median_mmd_index]
                median_mmd_loss_value = total_loss_list[median_mmd_index]
                median_mmd_similarity_value = similarity_matrix_list[median_mmd_index]
                ## max_index(validity_correction)
                val_cor_mmd = mmd_samples_value_list[max_index]
                val_cor_fgw = fgw_distance_list[max_index]
                val_cor_loss = total_loss_list[max_index]
                val_cor_similarity = similarity_matrix_list[max_index]
                ## total_loss_set(max)
                total_loss_list = list(total_loss_set)
                max_total_loss = max(total_loss_set)
                max_total_loss_index = total_loss_list.index(max_total_loss)
                maxloss_mmd = mmd_samples_value_list[max_total_loss_index]
                maxloss_fgw = fgw_distance_list[max_total_loss_index]
                maxloss_val_cor = val_cor_list[max_total_loss_index]
                maxloss_similarity = similarity_matrix_list[max_total_loss_index]
                ## total_loss_set(min)
                total_loss_list = list(total_loss_set)
                min_total_loss = min(total_loss_set)
                min_total_loss_index = total_loss_list.index(min_total_loss)
                minloss_mmd = mmd_samples_value_list[min_total_loss_index]
                minloss_fgw = fgw_distance_list[min_total_loss_index]
                minloss_val_cor = val_cor_list[min_total_loss_index]
                minloss_similarity = similarity_matrix_list[min_total_loss_index]
                ## total_loss_set(close to avg)
                n_value = 0
                total_value = 0
                for value in total_loss_list:
                    if isinstance(value, float) and not math.isnan(value):
                        total_value += value
                        n_value +=1
                average_loss = total_value / n_value
                min_diff = float('inf')  # 设置一个足够大的初始值
                # closest_value = 0
                for value in total_loss_list:
                    # print(value)
                    diff = abs(value - average_loss)
                    if diff < min_diff:
                        min_diff = diff
                        closest_value = value
                # print(total_loss_list)
                # print(closest_value)
                close_avg_total_loss_index = total_loss_list.index(closest_value)
                close_avg_total_loss_value = closest_value
                closeavgloss_mmd = mmd_samples_value_list[close_avg_total_loss_index]
                closeavgloss_fgw = fgw_distance_list[close_avg_total_loss_index]
                closeavgloss_val_cor = val_cor_list[close_avg_total_loss_index]
                closeavgloss_similarity = similarity_matrix_list[close_avg_total_loss_index]
                ## close to avg_mmd
                min_diff_value = float('inf')  # 设置一个足够大的初始值
                closest_avg_mmd_value = 0
                total_value_mmd = 0
                n_value_mmd = 0
                print(mmd_samples_value_list)
                for value_mmd in mmd_samples_value_list:
                    print(value_mmd)
                    if isinstance(value_mmd, torch.Tensor) and not math.isnan(value_mmd.item()):
                        total_value_mmd += value_mmd
                        n_value_mmd += 1
                print(n_value_mmd)
                avg_mmd_samples_value1 = total_value_mmd / n_value_mmd
                for value_mmd in mmd_samples_value_list:
                    diff1 = abs(value_mmd - avg_mmd_samples_value1)
                    if diff1 < min_diff_value:
                        min_diff_value = diff1
                        closest_avg_mmd_value = value_mmd
                closest_avg_mmd_index = mmd_samples_value_list.index(closest_avg_mmd_value)
                close_avg_mmd = closest_avg_mmd_value
                closeavgmmd_fgw = fgw_distance_list[closest_avg_mmd_index]
                closeavgmmd_val_cor = val_cor_list[closest_avg_mmd_index]
                closeavgmmd_loss = total_loss_list[closest_avg_mmd_index]
                closeavgmmd_similarity = similarity_matrix_list[closest_avg_mmd_index]
                ## close to avg_fgw
                min_diff_value = float('inf')  # 设置一个足够大的初始值
                closest_avg_fgw_value = 0
                for value_fgw in fgw_distance_list:
                    diff2 = abs(value_fgw - avg_fgw_distance_value)
                    if diff2 < min_diff_value:
                        min_diff_value = diff2
                        closest_avg_fgw_value = value_fgw
                closest_avg_fgw_index = fgw_distance_list.index(closest_avg_fgw_value)
                close_avg_fgw = fgw_distance_list[closest_avg_fgw_index]
                closeavgfgw_mmd = mmd_samples_value_list[closest_avg_fgw_index]
                closeavgfgw_val_cor = val_cor_list[closest_avg_fgw_index]
                closeavgfgw_loss = total_loss_list[closest_avg_fgw_index]
                closeavgfgw_similarity = similarity_matrix_list[closest_avg_fgw_index]
                ## random loss
                random_loss = random.choice(total_loss_set)
                random_loss_index = total_loss_list.index(random_loss)
                random_loss_mmd = mmd_samples_value_list[random_loss_index]
                random_loss_val_cor = val_cor_list[random_loss_index]
                random_loss_fgw = fgw_distance_list[random_loss_index]
                random_loss_similarity = similarity_matrix_list[random_loss_index]
                ## random mmd
                random_mmd = random.choice(mmd_samples_value_list)
                random_mmd_index = mmd_samples_value_list.index(random_mmd)
                random_mmd_loss = total_loss_list[random_mmd_index]
                random_mmd_val_cor = val_cor_list[random_mmd_index]
                random_mmd_fgw = fgw_distance_list[random_mmd_index]
                random_mmd_similarity = similarity_matrix_list[random_mmd_index]
                ## random fgw
                random_fgw = random.choice(fgw_distance_list)
                random_fgw_index = fgw_distance_list.index(random_fgw)
                random_fgw_loss = total_loss_list[random_fgw_index]
                random_fgw_val_cor = val_cor_list[random_fgw_index]
                random_fgw_mmd = mmd_samples_value_list[random_fgw_index]
                random_fgw_similarity = similarity_matrix_list[random_fgw_index]
                ## The first
                first_mmd = mmd_samples_value_list[0]
                first_fgw = fgw_distance_list[0]
                first_loss = total_loss_list[0]
                first_val_cor = val_cor_list[0]
                first_similarity = similarity_matrix_list[0]

                maxmmd_medianfgwmmd = max_mmd_samples_value + median_fgw_mmd_value
                maxmmd__medianfgwmmd = max_mmd_samples_value - median_fgw_mmd_value
                maxmmd___medianfgwmmd = (max_mmd_samples_value - median_fgw_mmd_value) / (max_mmd_samples_value + median_fgw_mmd_value)
                maxmmd____medianfgwmmd = max_mmd_samples_value * median_fgw_mmd_value
                maxmmd_____medianfgwmmd = max_mmd_samples_value / median_fgw_mmd_value
                maxmmd_10medianfgwmmd = 10 * max_mmd_samples_value + 10 * median_fgw_mmd_value
                maxmmd_5medianfgwmmd = 10 * max_mmd_samples_value + 5 * median_fgw_mmd_value
                maxmmd__5medianfgwmmd = 10 * max_mmd_samples_value - 5 * median_fgw_mmd_value
                maxmmd__13medianfgwmmd = 14 * max_mmd_samples_value - 13 * median_fgw_mmd_value

                maxmmd_medianmmd = max_mmd_samples_value + median_mmd_value
                maxmmd__medianmmd = max_mmd_samples_value - median_mmd_value
                maxmmd___medianmmd = (max_mmd_samples_value - median_mmd_value) / (max_mmd_samples_value + median_mmd_value)
                maxmmd____medianmmd = max_mmd_samples_value * median_mmd_value
                maxmmd_____medianmmd = max_mmd_samples_value / median_mmd_value
                maxmmd_10medianmmd = 10 * max_mmd_samples_value + 10 * median_mmd_value
                maxmmd_5medianmmd = 10 * max_mmd_samples_value + 5 * median_mmd_value
                maxmmd__5medianmmd = 10 * max_mmd_samples_value - 5 * median_mmd_value
                maxmmd__13medianmmd = 14 * max_mmd_samples_value - 13 * median_mmd_value

                # ##// 0.2 0.1 //
                # sqrt_median_mmd_value = math.sqrt((median_mmd_value-0.3)*(median_mmd_value-0.3))
                # sqrt_max_mmd_samples_value = math.sqrt((max_mmd_samples_value-0.5)*(max_mmd_samples_value-0.5))
                # maxmmd__cormedianmmd = (max_mmd_samples_value-0.5) * sqrt_median_mmd_value
                # maxmmd__cor2medianmmd = (max_mmd_samples_value-0.5) * sqrt_median_mmd_value + sqrt_max_mmd_samples_value * (median_mmd_value-0.3)
                # maxmmd__cor22medianmmd = ((max_mmd_samples_value-0.5) * sqrt_median_mmd_value + sqrt_max_mmd_samples_value * (median_mmd_value-0.3)) / (sqrt_median_mmd_value * sqrt_max_mmd_samples_value)
                # sqrt_max_mmd_samples_value2 = math.sqrt((max_mmd_samples_value - 0.53) * (max_mmd_samples_value - 0.53))
                # maxmmd__cormedianmmd2 = (max_mmd_samples_value - 0.53) * sqrt_median_mmd_value
                # maxmmd__cor2medianmmd2 = (max_mmd_samples_value - 0.53) * sqrt_median_mmd_value + sqrt_max_mmd_samples_value2 * (median_mmd_value - 0.3)
                # maxmmd__cor22medianmmd2 = ((max_mmd_samples_value - 0.53) * sqrt_median_mmd_value + sqrt_max_mmd_samples_value2 * (median_mmd_value - 0.3)) / (sqrt_median_mmd_value * sqrt_max_mmd_samples_value2)

                ## //0.3 0.1 0.1//
                sqrt_median_mmd_value = math.sqrt((median_mmd_value - 0.57) * (median_mmd_value - 0.57))
                sqrt_max_mmd_samples_value = math.sqrt((max_mmd_samples_value - 0.62) * (max_mmd_samples_value - 0.62))
                maxmmd__cormedianmmd = (max_mmd_samples_value - 0.62) * sqrt_median_mmd_value
                maxmmd__cor2medianmmd = (max_mmd_samples_value - 0.62) * sqrt_median_mmd_value + sqrt_max_mmd_samples_value * (median_mmd_value - 0.57)
                maxmmd__cor22medianmmd = ((max_mmd_samples_value - 0.62) * sqrt_median_mmd_value + sqrt_max_mmd_samples_value * (median_mmd_value - 0.57)) / (sqrt_median_mmd_value * sqrt_max_mmd_samples_value)
                sqrt_max_mmd_samples_value2 = math.sqrt((max_mmd_samples_value - 0.62) * (max_mmd_samples_value - 0.62))
                maxmmd__cormedianmmd2 = sqrt_max_mmd_samples_value + sqrt_median_mmd_value
                maxmmd__cor2medianmmd2 = (max_mmd_samples_value - 0.62) + (median_mmd_value - 0.57)
                maxmmd__cor22medianmmd2 =(max_mmd_samples_value - 0.62) + (median_mmd_value - 0.57)  / (sqrt_median_mmd_value * sqrt_max_mmd_samples_value2)

                maxmmd__04medianfgwmmd = 0.6 * max_mmd_samples_value - 0.4 * median_fgw_mmd_value
                maxmmd__03medianfgwmmd = 0.7 * max_mmd_samples_value - 0.3 * median_fgw_mmd_value
                maxmmd__02medianfgwmmd = 0.8 * max_mmd_samples_value - 0.2 * median_fgw_mmd_value
                maxmmd__add04medianfgwmmd = 0.6 * max_mmd_samples_value + 0.4 * median_fgw_mmd_value
                maxmmd__add03medianfgwmmd = 0.7 * max_mmd_samples_value + 0.3 * median_fgw_mmd_value
                maxmmd__add02medianfgwmmd = 0.8 * max_mmd_samples_value + 0.2 * median_fgw_mmd_value
                maxmmd__c04medianfgwmmd = 0.6 * max_mmd_samples_value * 0.4 * median_fgw_mmd_value
                maxmmd__c03medianfgwmmd = 0.7 * max_mmd_samples_value * 0.3 * median_fgw_mmd_value
                maxmmd__c02medianfgwmmd = 0.8 * max_mmd_samples_value * 0.2 * median_fgw_mmd_value
                maxmmd__ch04medianfgwmmd = 0.6 * max_mmd_samples_value / (0.4 * median_fgw_mmd_value)
                maxmmd__ch03medianfgwmmd = 0.7 * max_mmd_samples_value / (0.3 * median_fgw_mmd_value)
                maxmmd__ch02medianfgwmmd = 0.8 * max_mmd_samples_value / (0.2 * median_fgw_mmd_value)

                median_mmd_value2 = median_mmd_value*median_mmd_value
                max_mmd_samples_value2 = max_mmd_samples_value*max_mmd_samples_value
                max2_median2 = max_mmd_samples_value2 - median_mmd_value2
                max_median_fgw_value = median_mmd_value*max_mmd_samples_value - median_fgw_mmd_value*max_mmd_samples_value


                ############################################### STRATEGY ###############################################

                ## auc
                if args.task_type == 'regression':
                    rmse_aug_samples = mean_squared_error(y_true_np_samples, y_pred_aug_np_samples, squared=False)
                    rmse_ori_samples = mean_squared_error(y_true_np_samples, y_pred_ori_np_samples, squared=False)
                    perf = {'rmse_aug_samples': rmse_aug_samples, 'rmse_ori_samples': rmse_ori_samples}
                elif args.task_type == 'classification':
                    if args.dataset.startswith('ogbg'):
                        assert y_true_np_samples.shape[1] == y_pred_aug_np_samples.shape[1]
                        try:
                            auc_aug_samples = roc_auc_score(y_true_np_samples, y_pred_aug_np_samples)
                            auc_ori_samples = roc_auc_score(y_true_np_samples, y_pred_ori_np_samples)
                        except ValueError as e:
                            print(f"Error calculating AUC: {e}")
                            auc_aug_samples = None
                            auc_ori_samples = None
                    else:
                        assert y_true_np_samples.shape[1] == y_pred_aug_np_samples.shape[1]
                        try:
                            auc_aug_samples = _eval_rocauc(y_true_np_samples, y_pred_aug_np_samples)['rocauc']
                            auc_ori_samples = _eval_rocauc(y_true_np_samples, y_pred_ori_np_samples)['rocauc']
                            # auc_ori_aug_samples = _eval_rocauc(y_pred_aug_np_samples, y_pred_ori_np_samples)['rocauc']
                        except ValueError as e:
                            print(f"Error calculating AUC: {e}")
                            auc_aug_samples = None
                            auc_ori_samples = None
                            # auc_ori_aug_samples = None

                    # perf_auc = {'auc_aug_samples': auc_aug_samples, 'auc_ori_samples': auc_ori_samples, 'auc_ori_aug_samples': auc_ori_aug_samples}
                    perf_auc = {'auc_aug_samples': auc_aug_samples, 'auc_ori_samples': auc_ori_samples}

                maxmmd_aucaug = max_mmd_samples_value * auc_aug_samples
                medianmmd_aucaug = median_fgw_mmd_value * auc_aug_samples
                aa = maxmmd_medianfgwmmd * auc_aug_samples
                bb = maxmmd__medianfgwmmd * auc_aug_samples
                cc = maxmmd___medianfgwmmd * auc_aug_samples
                aa1 = maxmmd_medianfgwmmd * (1-auc_aug_samples)
                bb1 = maxmmd__medianfgwmmd * (1-auc_aug_samples)
                cc1 = maxmmd___medianfgwmmd * (1-auc_aug_samples)
                aa2 = maxmmd_medianfgwmmd * (-1 - auc_aug_samples)
                bb2 = maxmmd__medianfgwmmd * (-1 - auc_aug_samples)
                cc2 = maxmmd___medianfgwmmd * (-1 - auc_aug_samples)

                ## mean_squared_error, mean_absolute_error, r2_score
                rmse_aug_samples = mean_squared_error(y_true_np_samples, y_pred_aug_np_samples, squared=False)
                rmse_ori_samples = mean_squared_error(y_true_np_samples, y_pred_ori_np_samples, squared=False)
                rmse_ori_aug_samples = mean_squared_error(y_pred_aug_np_samples, y_pred_ori_np_samples, squared=False)
                mae_aug_samples = mean_absolute_error(y_true_np_samples, y_pred_aug_np_samples)
                mae_ori_samples = mean_absolute_error(y_true_np_samples, y_pred_ori_np_samples)
                mae_ori_aug_samples = mean_absolute_error(y_pred_aug_np_samples, y_pred_ori_np_samples)
                r2_aug_samples = r2_score(y_true_np_samples, y_pred_aug_np_samples)
                r2_ori_samples = r2_score(y_true_np_samples, y_pred_ori_np_samples)
                r2_ori_aug_samples = r2_score(y_pred_aug_np_samples, y_pred_ori_np_samples)
                r2_ori_samples_binary = r2_score(y_true_np_samples, y_pred_ori_np_samples_binary)
                l2_aug_samples = np.linalg.norm(y_true_np_samples - y_pred_aug_np_samples, ord=2)
                l2_ori_samples = np.linalg.norm(y_true_np_samples - y_pred_ori_np_samples, ord=2)
                l2_ori_aug_samples = np.linalg.norm(y_pred_ori_np_samples - y_pred_aug_np_samples, ord=2)
                perf_calculate_samples = {'rmse_aug_samples': rmse_aug_samples, 'rmse_ori_samples': rmse_ori_samples,
                                          'rmse_ori_aug_samples': rmse_ori_aug_samples,
                                          'mae_aug_samples': mae_aug_samples, 'mae_ori_samples': mae_ori_samples,
                                          'mae_ori_aug_samples': mae_ori_aug_samples,
                                          'r2_aug_samples': r2_aug_samples, 'r2_ori_samples': r2_ori_samples,
                                          'r2_ori_aug_samples': r2_ori_aug_samples,
                                          'r2_ori_samples_binary': r2_ori_samples_binary,
                                          'l2_aug_samples': l2_aug_samples, 'l2_ori_samples': l2_ori_samples,
                                          'l2_ori_aug_samples': l2_ori_aug_samples}

                ## 归一化 mean_squared_error, mean_absolute_error, r2_score
                with open('output_samples251114.txt', 'a') as file_samples:
                    # file_samples.write(f"--------------------------------------------------------------\n")
                    file_samples.write(f"te_batch_data：{te_batch_data}\n")
                    file_samples.write(f"The index of the step：{step}\n\n")

                    file_samples.write(f"mmd_samples_value_set：{mmd_samples_value_set}\n")
                    file_samples.write(f"avg_mmd_samples_value：{avg_mmd_samples_value}\n")
                    file_samples.write(f"max_mmd_samples_value：{max_mmd_samples_value}\n")
                    file_samples.write(f"min_mmd_samples_value：{min_mmd_samples_value}\n")
                    file_samples.write(f"max_min_mmd_samples_value：{max_min_mmd_samples_value}\n")

                    file_samples.write(f"max validity w/o correction：{max_validity_correction}\n")
                    file_samples.write(f"min validity w/o correction：{min_validity_correction}\n")
                    file_samples.write(f"Average validity w/o correction：{avg_validity_correction}\n")
                    file_samples.write(f"max_min validity w/o correction：{max_min_validity_correction}\n")

                    file_samples.write(f"avg_fgw_distance_value：{avg_fgw_distance_value}\n")
                    file_samples.write(f"max_fgw_distance_value：{max_fgw_distance_value}\n")
                    file_samples.write(f"min_fgw_distance_value：{min_fgw_distance_value}\n")
                    file_samples.write(f"max_min_fgw_distance_value：{max_min_fgw_distance_value}\n")

                    file_samples.write(f"median_fgw_value：{median_fgw_value}\n")
                    file_samples.write(f"median_fgw_mmd_value：{median_fgw_mmd_value}\n")
                    file_samples.write(f"median_fgw_val_cor_value：{median_fgw_val_cor_value}\n")
                    file_samples.write(f"median_fgw_loss_value：{median_fgw_loss_value}\n")

                    file_samples.write(f"median_mmd_value：{median_mmd_value}\n")
                    file_samples.write(f"median_mmd_fgw_value：{median_mmd_fgw_value}\n")
                    file_samples.write(f"median_mmd_val_cor_value：{median_mmd_val_cor_value}\n")
                    file_samples.write(f"median_mmd_loss_value：{median_mmd_loss_value}\n")

                    file_samples.write(f"val_cor_mmd：{val_cor_mmd}\n")
                    file_samples.write(f"val_cor_fgw：{val_cor_fgw}\n")
                    file_samples.write(f"val_cor_loss：{val_cor_loss}\n")

                    file_samples.write(f"max_total_loss：{max_total_loss}\n")
                    file_samples.write(f"maxloss_mmd：{maxloss_mmd}\n")
                    file_samples.write(f"maxloss_fgw：{maxloss_fgw}\n")
                    file_samples.write(f"maxloss_val_cor：{maxloss_val_cor}\n")

                    file_samples.write(f"min_total_loss：{min_total_loss}\n")
                    file_samples.write(f"minloss_mmd：{minloss_mmd}\n")
                    file_samples.write(f"minloss_fgw：{minloss_fgw}\n")
                    file_samples.write(f"minloss_val_cor：{minloss_val_cor}\n")

                    file_samples.write(f"average_loss：{average_loss}\n")
                    file_samples.write(f"close_avg_total_loss_value：{close_avg_total_loss_value}\n")
                    file_samples.write(f"closeavgloss_mmd：{closeavgloss_mmd}\n")
                    file_samples.write(f"closeavgloss_fgw：{closeavgloss_fgw}\n")
                    file_samples.write(f"closeavgloss_val_cor：{closeavgloss_val_cor}\n")

                    file_samples.write(f"close_avg_mmd：{close_avg_mmd}\n")
                    file_samples.write(f"closeavgmmd_fgw：{closeavgmmd_fgw}\n")
                    file_samples.write(f"closeavgmmd_val_cor：{closeavgmmd_val_cor}\n")
                    file_samples.write(f"closeavgmmd_loss：{closeavgmmd_loss}\n")

                    file_samples.write(f"close_avg_fgw：{close_avg_fgw}\n")
                    file_samples.write(f"closeavgfgw_mmd：{closeavgfgw_mmd}\n")
                    file_samples.write(f"closeavgfgw_val_cor：{closeavgfgw_val_cor}\n")
                    file_samples.write(f"closeavgfgw_loss：{closeavgfgw_loss}\n")

                    file_samples.write(f"random_loss：{random_loss}\n")
                    file_samples.write(f"random_loss_mmd：{random_loss_mmd}\n")
                    file_samples.write(f"random_loss_val_cor：{random_loss_val_cor}\n")
                    file_samples.write(f"random_loss_fgw：{random_loss_fgw}\n")
                    file_samples.write(f"random_mmd：{random_mmd}\n")
                    file_samples.write(f"random_mmd_loss：{random_mmd_loss}\n")
                    file_samples.write(f"random_mmd_val_cor：{random_mmd_val_cor}\n")
                    file_samples.write(f"random_mmd_fgw：{random_mmd_fgw}\n")
                    file_samples.write(f"random_fgw：{random_fgw}\n")
                    file_samples.write(f"random_fgw_loss：{random_fgw_loss}\n")
                    file_samples.write(f"random_fgw_val_cor：{random_fgw_val_cor}\n")
                    file_samples.write(f"random_fgw_mmd：{random_fgw_mmd}\n")

                    file_samples.write(f"median_mmd_similarity_value：{median_mmd_similarity_value}\n")
                    file_samples.write(f"val_cor_similarity：{val_cor_similarity}\n")
                    file_samples.write(f"maxloss_similarity：{maxloss_similarity}\n")
                    file_samples.write(f"minloss_similarity：{minloss_similarity}\n")
                    file_samples.write(f"closeavgloss_similarity：{closeavgloss_similarity}\n")
                    file_samples.write(f"closeavgmmd_similarity：{closeavgmmd_similarity}\n")
                    file_samples.write(f"closeavgfgw_similarity：{closeavgfgw_similarity}\n")
                    file_samples.write(f"random_loss_similarity：{random_loss_similarity}\n")
                    file_samples.write(f"random_mmd_similarity：{random_mmd_similarity}\n")
                    file_samples.write(f"random_fgw_similarity：{random_fgw_similarity}\n")
                    file_samples.write(f"first_mmd：{first_mmd}\n")
                    file_samples.write(f"first_fgw：{first_fgw}\n")
                    file_samples.write(f"first_loss：{first_loss}\n")
                    file_samples.write(f"first_val_cor：{first_val_cor}\n")
                    file_samples.write(f"first_similarity：{first_similarity}\n")

                    file_samples.write(f"max2_median2：{max2_median2}\n")
                    file_samples.write(f"max_median_fgw_value：{max_median_fgw_value}\n")

                    file_samples.write(f"maxmmd_medianfgwmmd：{maxmmd_medianfgwmmd}\n")
                    file_samples.write(f"maxmmd__medianfgwmmd：{maxmmd__medianfgwmmd}\n")
                    file_samples.write(f"maxmmd___medianfgwmmd：{maxmmd___medianfgwmmd}\n")
                    file_samples.write(f"maxmmd____medianfgwmmd：{maxmmd____medianfgwmmd}\n")
                    file_samples.write(f"maxmmd_____medianfgwmmd：{maxmmd_____medianfgwmmd}\n")
                    file_samples.write(f"maxmmd_aucaug：{maxmmd_aucaug}\n")
                    file_samples.write(f"medianmmd_aucaug：{medianmmd_aucaug}\n")
                    file_samples.write(f"maxmmd_10medianfgwmmd：{maxmmd_10medianfgwmmd}\n")
                    file_samples.write(f"maxmmd_5medianfgwmmd：{maxmmd_5medianfgwmmd}\n")
                    file_samples.write(f"maxmmd__5medianfgwmmd：{maxmmd__5medianfgwmmd}\n")
                    file_samples.write(f"maxmmd__13medianfgwmmd：{maxmmd__13medianfgwmmd}\n")

                    file_samples.write(f"maxmmd_medianmmd：{maxmmd_medianmmd}\n")
                    file_samples.write(f"maxmmd__medianmmd：{maxmmd__medianmmd}\n")
                    file_samples.write(f"maxmmd___medianmmd：{maxmmd___medianmmd}\n")
                    file_samples.write(f"maxmmd____medianmmd：{maxmmd____medianmmd}\n")
                    file_samples.write(f"maxmmd_____medianmmd：{maxmmd_____medianmmd}\n")
                    file_samples.write(f"maxmmd_10medianmmd：{maxmmd_10medianmmd}\n")
                    file_samples.write(f"maxmmd_5medianmmd：{maxmmd_5medianmmd}\n")
                    file_samples.write(f"maxmmd__5medianmmd：{maxmmd__5medianmmd}\n")
                    file_samples.write(f"maxmmd__13medianmmd：{maxmmd__13medianmmd}\n")

                    file_samples.write(f"maxmmd__04medianfgwmmd：{maxmmd__04medianfgwmmd}\n")
                    file_samples.write(f"maxmmd__03medianfgwmmd：{maxmmd__03medianfgwmmd}\n")
                    file_samples.write(f"maxmmd__02medianfgwmmd：{maxmmd__02medianfgwmmd}\n")
                    file_samples.write(f"maxmmd__add04medianfgwmmd：{maxmmd__add04medianfgwmmd}\n")
                    file_samples.write(f"maxmmd__add03medianfgwmmd：{maxmmd__add03medianfgwmmd}\n")
                    file_samples.write(f"maxmmd__add02medianfgwmmd：{maxmmd__add02medianfgwmmd}\n")
                    file_samples.write(f"maxmmd__c04medianfgwmmd：{maxmmd__c04medianfgwmmd}\n")
                    file_samples.write(f"maxmmd__c03medianfgwmmd：{maxmmd__c03medianfgwmmd}\n")
                    file_samples.write(f"maxmmd__c02medianfgwmmd：{maxmmd__c02medianfgwmmd}\n")
                    file_samples.write(f"maxmmd__ch04medianfgwmmd：{maxmmd__ch04medianfgwmmd}\n")
                    file_samples.write(f"maxmmd__ch03medianfgwmmd：{maxmmd__ch03medianfgwmmd}\n")
                    file_samples.write(f"maxmmd__ch02medianfgwmmd：{maxmmd__ch02medianfgwmmd}\n")

                    file_samples.write(f"maxmmd__cormedianmmd：{maxmmd__cormedianmmd}\n")
                    file_samples.write(f"maxmmd__cor2medianmmd：{maxmmd__cor2medianmmd}\n")
                    file_samples.write(f"maxmmd__cor22medianmmd：{maxmmd__cor22medianmmd}\n")
                    file_samples.write(f"maxmmd__cormedianmmd2：{maxmmd__cormedianmmd2}\n")
                    file_samples.write(f"maxmmd__cor2medianmmd2：{maxmmd__cor2medianmmd2}\n")
                    file_samples.write(f"maxmmd__cor22medianmmd2：{maxmmd__cor22medianmmd2}\n")

                    file_samples.write(f"auc_ori_samples：{auc_ori_samples}\n")
                    file_samples.write(f"auc_aug_samples：{auc_aug_samples}\n")
                    file_samples.write(f"-----------------------------------------------------------------------------------------------------------------\n")
                file_samples.close()
                print("已添加至output_samples.txt文件中。")
                # assert False


            #         ######################################  添加数据选择策略mmd  ##########################################
            #         ## 添加数据选择策略mmd(计算全部)
            #         te_kernel_matrix = guassian_kernel_ma(te_batch_data.x, te_batch_data.x)
            #         aug_kernel_matrix = guassian_kernel_ma(aug_new_data.x, aug_new_data.x)
            #         te_batch_kernel_mean = torch.mean(te_kernel_matrix)
            #         aug_batch_kernel_mean = torch.mean(aug_kernel_matrix)
            #         mmd_value = (te_batch_kernel_mean + aug_batch_kernel_mean - 2 * torch.mean(guassian_kernel_ma(te_batch_data.x, aug_new_data.x)))
            #         mmd_value_sum = mmd_value_sum + mmd_value
            #         print("mmd_value值为", mmd_value)
            #         count_mmd = count_mmd + 1
            #         ## 将过程中最大和最小的参数记录下来
            #         if (te_batch_kernel_mean + aug_batch_kernel_mean - 2 * torch.mean(guassian_kernel_ma(te_batch_data.x, aug_new_data.x))) > max_mmd:
            #             max_mmd = (te_batch_kernel_mean + aug_batch_kernel_mean - 2 * torch.mean(guassian_kernel_ma(te_batch_data.x, aug_new_data.x)))
            #         if (te_batch_kernel_mean + aug_batch_kernel_mean - 2 * torch.mean(guassian_kernel_ma(te_batch_data.x, aug_new_data.x))) < min_mmd:
            #             min_mmd = (te_batch_kernel_mean + aug_batch_kernel_mean - 2 * torch.mean(guassian_kernel_ma(te_batch_data.x, aug_new_data.x)))
            #
            #         # ## 添加数据选择策略mmd(计算随机)
            #         # t = random.uniform(0, 1)
            #         # if count==0 and t>0.5:
            #         #     te_kernel_matrix = guassian_kernel_ma(te_batch_data.x, te_batch_data.x)
            #         #     aug_kernel_matrix = guassian_kernel_ma(aug_new_data.x, aug_new_data.x)
            #         #     te_batch_kernel_mean = torch.mean(te_kernel_matrix)
            #         #     aug_batch_kernel_mean = torch.mean(aug_kernel_matrix)
            #         #     mmd_value_random = te_batch_kernel_mean + aug_batch_kernel_mean - 2 * torch.mean(guassian_kernel_ma(te_batch_data.x, aug_new_data.x))
            #         #     count = 1
            #         # # print(mmd_value_random)
            #
            #     ## 添加数据选择策略mmd(计算全部)
            #     mmd_value_mean = mmd_value_sum/20
            #     t = mmd_value_mean
            #     # count += 1
            #     print(1)
            #     print("mmd_value_mean值为", mmd_value_mean)
            #     print(2)
            #     print("最大值为", max_mmd, "最小值为", min_mmd)
            #     print("count_mmd为", count_mmd)
            #     print(3)
            #
            #     # ## 添加数据选择策略(计算随机)
            #     # print(mmd_value_random)
            #
            # ## 添加数据选择策略mmd(计算全部)
            # # print("随机取值为", mmd_value_random)
            # ##########################################################################################################

            ###################################### validity_correction ###############################################

            max_validity_loss = total_loss_set[max_index]
            max_validity_index = int((max_index + 1) / sample_steps_value)
            if len(args.p_ratio_lists) > 1:
                logging.info('Selected min_negative_loss = {} in {}-th p_ratio = {}'.format(max_validity_loss,
                                                                                            max_index + 1,
                                                                                            args.p_ratio_lists[
                                                                                                max_validity_index]))
            elif len(args.sample_steps) > 1:
                logging.info('Selected min_negative_loss = {} in {}-th sample_steps = {}'.format(max_validity_loss,
                                                                                                 max_index + 1,
                                                                                                 args.sample_steps[
                                                                                                     max_validity_index]))
            aug_new_data_out = aug_new_data_set[max_index]
            ## 计算max_validity_correction current_index为max_index时的MMD
            te_kernel_matrix = guassian_kernel_ma(te_batch_data.x, te_batch_data.x)
            aug_kernel_matrix = guassian_kernel_ma(aug_new_data_out.x, aug_new_data_out.x)
            te_batch_kernel_mean = torch.mean(te_kernel_matrix)
            aug_batch_kernel_mean = torch.mean(aug_kernel_matrix)
            mmd_value = (te_batch_kernel_mean + aug_batch_kernel_mean - 2 * torch.mean(
                guassian_kernel_ma(te_batch_data.x, aug_new_data_out.x)))


            with torch.no_grad():
                y_pred_test_aug_, _, te_aug_gembs = model(aug_new_data_out)  ## 使用model来生成预测(增强数据后的预测结果)

                if args.task_type == 'regression':
                    y_pred_test_aug = y_pred_test_aug_.detach().cpu()
                    y_pred_test_ori = te_y_pred.detach().cpu()
                    average_gemb = args.te_aug_ratio * te_aug_gembs + (1 - args.te_aug_ratio) * te_gembs
                    y_pred_test_aug_avg = model.predictor(average_gemb).detach().cpu()

                    y_ground_true = te_batch_data.y.detach().cpu()
                    y_pred_test_aug_avg_ls.append(y_pred_test_aug_avg)
                elif args.task_type == 'classification':
                    y_pred_test_aug = y_pred_test_aug_.detach().cpu()  ## y_pred_test_aug_ 是增强数据后的预测结果
                    y_pred_test_ori = te_y_pred.detach().cpu()
                    ensemble_result = ensemble_predictions(y_pred_test_aug, y_pred_test_ori)  ## ensemble_predictions 函数用于组合不同样本的预测结果
                    y_ground_true = te_batch_data.y.detach().cpu()
                    y_pred_test_aug_avg_ls.append(ensemble_result)
                    ## y_pred_test_aug
                    y_pred_test_aug_binary1 = F.softmax(y_pred_test_aug_, dim=1).detach().cpu()  ## F.softmax

                    y_pred_test_aug_prob = torch.exp(y_pred_test_aug_)
                    # 归一化概率分布使其和为1
                    y_pred_test_aug_prob /= torch.sum(y_pred_test_aug_prob, dim=1, keepdim=True)
                    # 使用argmax获取预测的类别
                    y_pred_test_aug__ = torch.round(y_pred_test_aug_prob)
                    y_pred_test_aug_binary2 = y_pred_test_aug__.detach().cpu()
                    ## y_pred_test_ori
                    y_pred_test_ori_binary1 = F.softmax(te_y_pred, dim=1).detach().cpu()  ## F.softmax

                    y_pred_test_ori_prob = torch.exp(te_y_pred)
                    # 归一化概率分布使其和为1
                    y_pred_test_ori_prob /= torch.sum(y_pred_test_ori_prob, dim=1, keepdim=True)
                    # 使用argmax获取预测的类别
                    y_pred_test_ori__ = torch.round(y_pred_test_ori_prob)
                    y_pred_test_ori_binary2 = y_pred_test_ori__.detach().cpu()

                y_ground_true_ls.append(y_ground_true)
                y_pred_test_aug_ls.append(y_pred_test_aug)
                y_pred_test_ori_ls.append(y_pred_test_ori)
                mmd_value_set.append(mmd_value)

                y_pred_test_aug_ls_binary2.append(y_pred_test_aug_binary2)  ##
                y_pred_test_ori_ls_binary2.append(y_pred_test_ori_binary2)  ##
                y_pred_test_aug_ls_binary1.append(y_pred_test_aug_binary1)  ##
                y_pred_test_ori_ls_binary1.append(y_pred_test_ori_binary1)

                # y_pred_test_aug_ls_binary2.append(y_pred_test_aug_binary2)  ##
                # y_pred_test_ori_ls_binary2.append(y_pred_test_ori_binary2)  ##
                # y_pred_test_aug_ls_binary1.append(y_pred_test_aug_binary1)  ##
                # y_pred_test_ori_ls_binary1.append(y_pred_test_ori_binary1)

    y_true_np = torch.cat(y_ground_true_ls, dim=0).numpy()  ## 预测结果转换为numpy数组，以便进行后续分析或存储  (for te_batch_data)
    y_pred_aug_np = torch.cat(y_pred_test_aug_ls, dim=0).numpy()
    y_pred_aug_avg_np = torch.cat(y_pred_test_aug_avg_ls, dim=0).numpy()
    y_pred_ori_np = torch.cat(y_pred_test_ori_ls, dim=0).numpy()

    y_pred_aug_np_binary = torch.cat(y_pred_test_aug_ls_binary1, dim=0).numpy()  ## 为了计算acc
    y_pred_ori_np_binary = torch.cat(y_pred_test_ori_ls_binary1, dim=0).numpy()  ## 为了计算acc
    # y_pred_aug_np_binary = torch.cat(y_pred_test_aug_ls_binary1, dim=0).numpy()  ## 为了计算acc
    # y_pred_ori_np_binary = torch.cat(y_pred_test_ori_ls_binary1, dim=0).numpy()  ## 为了计算acc

    # print("test_data_add:", test_data_add)
    # print(y_pred_ori_np)
    # print("y_pred_ori_np元素数量为：", len(y_pred_ori_np))
    # print("y_true_np元素数量为：", len(y_true_np))
    # print("y_pred_aug_np元素数量为：", len(y_pred_aug_np))

    ## Average MMD
    avg_mmd_value = sum(mmd_value_set) / len(mmd_value_set)
    max_mmd_value = max(mmd_value_set)
    min_mmd_value = min(mmd_value_set)
    max_min_mmd_value = max(mmd_value_set) - min(mmd_value_set)
    print(test_data_add)  ## 380 (19*20)

    with open('output_samples251114.txt', 'a') as file_samples:
        file_samples.write(f"==============================================================\n")
        file_samples.write(f"te_batch_data：{te_batch_data}\n")
        file_samples.write(f"mmd_value_set：{mmd_value_set}\n")
        file_samples.write(f"avg_mmd_value：{avg_mmd_value}\n")
        file_samples.write(f"max_mmd_value：{max_mmd_value}\n")
        file_samples.write(f"min_mmd_value：{min_mmd_value}\n")
        file_samples.write(f"max_min_mmd_value：{max_min_mmd_value}\n")
        file_samples.write(f"max validity w/o correction：{max_validity_correction}\n")
        # file.write(f"min validity w/o correction：{min_validity_correction}\n")
        # file.write(f"Average validity w/o correction：{avg_validity_correction}\n")
        # file.write(f"max_min validity w/o correction：{max_min_validity_correction}\n")
        # file.write(f"avg_fgw_distance_value：{avg_fgw_distance_value}\n")
        # file.write(f"max_fgw_distance_value：{max_fgw_distance_value}\n")
        # file.write(f"min_fgw_distance_value：{min_fgw_distance_value}\n")
        # file.write(f"max_min_fgw_distance_value：{max_min_fgw_distance_value}\n")
        file_samples.write(f"==============================================================\n")
    file_samples.close()

    # return y_true_np, y_pred_aug_np, y_pred_aug_avg_np, y_pred_ori_np, y_pred_aug_np_binary, y_pred_ori_np_binary
    return y_true_np, y_pred_aug_np, y_pred_aug_avg_np, y_pred_ori_np

