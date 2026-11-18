import os
from time import time
import numpy as np
import networkx as nx

import torch
from torch.utils.data import Dataset
import json
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random
### Code adapted from GraphEBM
def load_mol(filepath):
    print(f'Loading file {filepath}')
    if not os.path.exists(filepath):
        raise ValueError(f'Invalid filepath {filepath} for dataset')
    load_data = np.load(filepath)
    result = []
    i = 0
    while True:
        key = f'arr_{i}'
        if key in load_data.keys():
            result.append(load_data[key])
            i += 1
        else:
            break

    #return list(map(lambda x, a: (x, a), result[0], result[1]))
    return list(map(lambda x, a, y: (x, a, y), result[0], result[1], result[2]))


class MolDataset(Dataset):
    def __init__(self, mols, transform):
        self.mols = mols
        self.transform = transform

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        return self.transform(self.mols[idx])


def get_transform_fn(dataset):
    if dataset == 'QM9':
        def transform(data):
            x, adj = data
            # the last place is for virtual nodes
            # 6: C, 7: N, 8: O, 9: F
            x_ = np.zeros((9, 5))
            indices = np.where(x >= 6, x - 6, 4)
            x_[np.arange(9), indices] = 1
            x = torch.tensor(x_).to(torch.float32)
            # single, double, triple and no-bond; the last channel is for virtual edges
            adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                                    axis=0).astype(np.float32)

            x = x[:, :-1]                               # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
            adj = torch.tensor(adj.argmax(axis=0))      # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
            # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
            return x, adj

    elif dataset == 'ZINC250k':
        def transform(data):
            x, adj = data
            # the last place is for virtual nodes
            # 6: C, 7: N, 8: O, 9: F, 15: P, 16: S, 17: Cl, 35: Br, 53: I
            zinc250k_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
            x_ = np.zeros((38, 10), dtype=np.float32)
            for i in range(38):
                ind = zinc250k_atomic_num_list.index(x[i])
                x_[i, ind] = 1.
            x = torch.tensor(x_).to(torch.float32)
            # single, double, triple and no-bond; the last channel is for virtual edges
            adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                                 axis=0).astype(np.float32)

            x = x[:, :-1]                               # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
            adj = torch.tensor(adj.argmax(axis=0))      # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
            # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
            return x, adj

    elif dataset == 'ogbg-molfreesolv':
        def transform(data):
            x, adj, y = data
            # the last place is for virtual nodes
            # 6: C, 7: N, 8: O, 9: F, 15: P, 16: S, 17: Cl, 35: Br, 53: I
            ogbg_molfreesolv_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
            x_ = np.zeros((24, 10), dtype=np.float32)
            for i in range(24):
                ind = ogbg_molfreesolv_atomic_num_list.index(x[i])
                x_[i, ind] = 1.
            x = torch.tensor(x_).to(torch.float32)
            # single, double, triple and no-bond; the last channel is for virtual edges
            adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                                 axis=0).astype(np.float32)
            bond_type_num = adj.shape[0] - 1

            x = x[:, :-1]                               # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
            adj = torch.tensor(adj.argmax(axis=0))      # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
            # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
            bonds = adj[adj > 0].long() - 1  # Adjust index for one-hot encoding
            edge_attr = torch.zeros(bonds.numel(), bond_type_num).to(torch.float32)
            edge_attr[torch.arange(bonds.numel()), bonds] = 1
            return x, adj, edge_attr, y

    elif dataset == 'ogbg-molbace':
        def transform(data):
            x, adj, y = data
            # the last place is for virtual nodes
            ogbg_molbace_atomic_num_list = [6, 7, 8, 9, 16, 17, 35, 53, 0]
            x_ = np.zeros((97, 9), dtype=np.float32)
            for i in range(97):
                ind = ogbg_molbace_atomic_num_list.index(x[i])
                x_[i, ind] = 1.
            x = torch.tensor(x_).to(torch.float32)
            # single, double, triple and no-bond; the last channel is for virtual edges
            adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                                 axis=0).astype(np.float32)
            bond_type_num = adj.shape[0] - 1

            x = x[:, :-1]                               # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
            adj = torch.tensor(adj.argmax(axis=0))      # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
            # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
            # Create one-hot vectors for each edge
            # Flatten the matrix and filter out zero (no-bond/virtual) indices
            bonds = adj[adj > 0].long()-1  # Adjust index for one-hot encoding
            edge_attr = torch.zeros(bonds.numel(), bond_type_num).to(torch.float32)
            edge_attr[torch.arange(bonds.numel()), bonds] = 1
            return x, adj, edge_attr, y

    elif dataset == 'ogbg-molbbbp':
        def transform(data):
            x, adj, y = data
            # the last place is for virtual nodes
            # 6: C, 7: N, 8: O, 9: F, 15: P, 16: S, 17: Cl, 35: Br, 53: I
            ogbg_molbbbp_atomic_num_list = [17, 6, 7, 8, 9, 16, 35, 53, 1, 11, 15, 20, 5, 0]
            x_ = np.zeros((132, 14), dtype=np.float32)
            for i in range(132):
                ind = ogbg_molbbbp_atomic_num_list.index(x[i])
                x_[i, ind] = 1.
            x = torch.tensor(x_).to(torch.float32)
            # single, double, triple and no-bond; the last channel is for virtual edges
            adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                                 axis=0).astype(np.float32)

            bond_type_num = adj.shape[0] - 1
            x = x[:, :-1]                               # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
            adj = torch.tensor(adj.argmax(axis=0))      # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
            # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
            bonds = adj[adj > 0].long() - 1  # Adjust index for one-hot encoding
            edge_attr = torch.zeros(bonds.numel(), bond_type_num).to(torch.float32)
            edge_attr[torch.arange(bonds.numel()), bonds] = 1
            return x, adj, edge_attr, y

    elif dataset == 'ogbg-molclintox':
        def transform(data):
            x, adj, y = data
            # the last place is for virtual nodes
            # 6: C, 7: N, 8: O, 9: F, 15: P, 16: S, 17: Cl, 35: Br, 53: I
            ogbg_molclintox_atomic_num_list = [6, 17, 8, 1, 7, 43, 15, 9, 16, 34, 5, 26, 13, 35, 53, 20, 78, 83, 79, 81,
                                               24, 29, 25, 30, 14, 80, 33, 22, 0]
            x_ = np.zeros((136, 29), dtype=np.float32)
            for i in range(136):
                ind = ogbg_molclintox_atomic_num_list.index(x[i])
                x_[i, ind] = 1.
            x = torch.tensor(x_).to(torch.float32)
            # single, double, triple and no-bond; the last channel is for virtual edges
            adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                                 axis=0).astype(np.float32)

            bond_type_num = adj.shape[0] - 1
            x = x[:, :-1]  # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
            adj = torch.tensor(adj.argmax(axis=0))  # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
            # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
            bonds = adj[adj > 0].long() - 1  # Adjust index for one-hot encoding
            edge_attr = torch.zeros(bonds.numel(), bond_type_num).to(torch.float32)
            edge_attr[torch.arange(bonds.numel()), bonds] = 1
            return x, adj, edge_attr, y

    return transform


def dataloader(args, get_graph_list=False):
    start_time = time()

    mols = load_mol(os.path.join(args.data_dir, f'{args.dataset.lower()}_kekulized.npz'))

    with open(os.path.join(args.data_dir, f'valid_idx_{args.dataset.lower()}.json')) as f:
        test_idx = json.load(f)

    if args.dataset == 'QM9':
        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]

    elif args.dataset == 'ogbg-molfreesolv' or args.dataset == 'ogbg-molbace' or args.dataset == 'ogbg-molbbbp' or args.dataset == 'ogbg-molclintox':
        with open(os.path.join(args.data_dir, f'train_idx_{args.dataset.lower()}.json')) as f1:
            train_idx = json.load(f1)
    # train_idx = [i for i in range(len(mols)) if i not in test_idx]
    print(f'Number of training mols: {len(train_idx)} | Number of test mols: {len(test_idx)}')
    #train_idx = [i for i in range(len(mols)) if i not in test_idx and i not in val_idx]
    train_idx = [i for i in range(len(mols)) if i not in test_idx]
    #print(f'Number of training mols: {len(train_idx)}| Number of val mols: {len(val_idx)} | Number of test mols: {len(test_idx)}')

    train_mols = [mols[i] for i in train_idx]
    test_mols = [mols[i] for i in test_idx]
    #val_mols = [mols[i] for i in val_idx]

    train_dataset = MolDataset(train_mols, get_transform_fn(args.dataset))
    test_dataset = MolDataset(test_mols, get_transform_fn(args.dataset))
    val_dataset = MolDataset(val_mols, get_transform_fn(args.dataset))

    if get_graph_list:
        train_mols_nx = [nx.from_numpy_array(np.array(adj)) for x, adj in train_dataset]
        test_mols_nx = [nx.from_numpy_array(np.array(adj)) for x, adj in test_dataset]
        return train_mols_nx, test_mols_nx

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    print(f'{time() - start_time:.2f} sec elapsed for data loading')
    return train_dataloader, test_dataloader


def dataloader_tt(args):
    start_time = time()

    if args.dataset == 'ogbg-molbace':
        mols = load_mol(os.path.join(args.data_dir, f'{args.dataset.lower()}_kekulized_Xinz_0731.npz'))
    elif args.dataset == 'ogbg-molbbbp':
        mols = load_mol(os.path.join(args.data_dir, f'{args.dataset.lower()}_kekulized_xinz_0829.npz'))
    elif args.dataset == 'ogbg-molclintox':
        mols = load_mol(os.path.join(args.data_dir, f'{args.dataset.lower()}_kekulized_xin.npz'))
    else:
        mols = load_mol(os.path.join(args.data_dir, f'{args.dataset.lower()}_kekulized.npz'))

    with open(os.path.join(args.data_dir, f'valid_idx_{args.dataset.lower()}.json')) as f:
        test_idx = json.load(f)

    #with open(os.path.join(args.data_dir, f'val_idx_{args.dataset.lower()}.json')) as f:
    #    val_idx = json.load(f)

    if args.dataset == 'QM9':
        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]
        train_idx = [i for i in range(len(mols)) if i not in test_idx and i not in val_idx]
    elif args.dataset == 'ogbg-molfreesolv' or args.dataset == 'ogbg-molbace' or args.dataset == 'ogbg-molbbbp' or args.dataset == 'ogbg-molclintox':
        with open(os.path.join(args.data_dir, f'train_idx_{args.dataset.lower()}.json')) as f1:
            train_idx = json.load(f1)

    val_file_path = os.path.join('./data', f'val_idx_{args.dataset}.json')
    if os.path.exists(val_file_path):
        with open(os.path.join('./data', f'val_idx_{args.dataset}.json')) as f:
            val_idx = json.load(f)
            print('Loading the validation set index from = ', val_file_path)
    else:
        val_idx = random.sample(train_idx, len(test_idx))
        # Save val_idx to a JSON file
        with open(val_file_path, 'w') as f:
            json.dump(val_idx, f)
        print('The validation set index saved to ', val_file_path)
        assert False

    print(
        f'Number of training mols: {len(train_idx)}| Number of val mols: {len(val_idx)} | Number of test mols: {len(test_idx)}')

    train_mols = [mols[i] for i in train_idx]
    test_mols = [mols[i] for i in test_idx]
    val_mols = [mols[i] for i in val_idx]

    train_dataset = MolDataset(train_mols, get_transform_fn(args.dataset))
    test_dataset = MolDataset(test_mols, get_transform_fn(args.dataset))
    val_dataset = MolDataset(val_mols, get_transform_fn(args.dataset))

    train_data_list = create_data_list(train_dataset)
    test_data_list = create_data_list(test_dataset)
    val_data_list = create_data_list(val_dataset)

    ## ogbg-molfreesolv
    if args.dataset == 'ogbg-molfreesolv':
        train_dataloader = DataLoader(train_data_list, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data_list, batch_size=10, shuffle=False)
        val_dataloader = DataLoader(val_data_list, batch_size=args.batch_size, shuffle=True)
        print(f'{time() - start_time:.2f} sec elapsed for data loading')
        return train_dataloader, test_dataloader, val_dataloader

    # ## ogbg-molclintox   导致 main_test_aug_obj.py "model.load_state_dict(torch.load(args.pretrain_model_path))"参数形状不匹配--(num_taske改为2)
    # if args.dataset == 'ogbg-molclintox':    # output_samples1018clintox-0.1 0.1 ori.txt
    #     # Use DataLoader on the list of Data objects
    #     train_dataloader = DataLoader(train_data_list, batch_size=args.batch_size, shuffle=True)
    #     # test_dataloader = DataLoader(test_data_list, batch_size=args.batch_size, shuffle=False)
    #     # test_dataloader = DataLoader(test_data_list, batch_size=10, shuffle=True)
    #     test_dataloader = DataLoader(test_data_list, batch_size=10, shuffle=False)
    #     val_dataloader = DataLoader(val_data_list, batch_size=args.batch_size, shuffle=True)
    #     print(f'{time() - start_time:.2f} sec elapsed for data loading')
    #     return train_dataloader, test_dataloader, val_dataloader

    ## ogbg-molclintox   导致 main_test_aug_obj.py "model.load_state_dict(torch.load(args.pretrain_model_path))"参数形状不匹配--(num_taske改为2)
    if args.dataset == 'ogbg-molclintox':
        ## 按照label存数据
        num_labels = len(test_data_list)
        # print(test_data_list)    ## [Data(x=[97, 8], edge_index=[2, 70], edge_attr=[70, 3], y=[1, 1]), Data(x=[97, 8], edge_index=[2, 100], edge_attr=[100, 3], y=[1, 1]), ...
        # data_y = [data.y for data in test_data_list]
        # print(data_y)
        # data_y0 = [data[0, 0] for data in data_y]  ## 取第一个元素
        # print(data_y0)
        # data_y1 = [data[0, 1] for data in data_y]  ## 取第二个元素
        # print(data_y1)
        num_zeros = sum(1 for data in test_data_list if data.y[0, 0] == 0)
        num_ones = num_labels - num_zeros
        print(num_zeros,num_ones)  ## data.y[0, 0]: 9 139    data.y[0, 1]: 138 10
        # assert False
        num_batches = num_zeros  ## 组数 9
        if num_labels % num_batches == 0:
            test_batch_size = num_labels / num_batches
        else:
            test_batch_size = (num_labels-(num_labels-(num_labels//(num_batches-1))*(num_batches-1)))//(num_batches-1)

        # 将标签为0和1的样本分别存储起来
        zeros_data = [data for data in test_data_list if data.y[0, 0] == 0]
        ones_data = [data for data in test_data_list if data.y[0, 0] == 1]

        # 初始化批次列表
        batches = []
        batch_pairs = [[] for _ in range(num_batches)]
        pairs = [0 for _ in range(num_batches)]  ## counting
        seed = 42
        # 为每个批次随机选择一个标签为0和一个标签为1的样本
        for i in range(num_batches):
            # 如果还有标签为0的样本，随机选择一个
            random.seed(seed)
            if zeros_data:
                zero_sample = random.choice(zeros_data)
                zeros_data.remove(zero_sample)  # 从列表中移除已选中的样本
                batch_pairs[i].append(zero_sample)
                pairs[i] = pairs[i] + 1
            else:
                zero_sample = None

            # 如果还有标签为1的样本，随机选择一个
            if ones_data:
                one_sample = random.choice(ones_data)
                ones_data.remove(one_sample)  # 从列表中移除已选中的样本
                batch_pairs[i].append(one_sample)
                pairs[i] = pairs[i] + 1
            else:
                one_sample = None

        # for i in range(num_batches):
        #     data_y = [data.y for data in batch_pairs[i]]
        #     print(data_y)
        # assert False

        # 向每个批次中添加样本，直到达到最大批次大小或所有样本都被使用   (如果没添加到)
        if num_labels % test_batch_size == 0:
            while zeros_data or ones_data:
                for i in range(num_batches):
                    # 如果还有标签为0的样本，随机选择一个
                    random.seed(seed)
                    if zeros_data and pairs[i] < test_batch_size:
                        zero_sample = random.choice(zeros_data)
                        zeros_data.remove(zero_sample)  # 从列表中移除已选中的样本
                        batch_pairs[i].append(zero_sample)
                        pairs[i] = pairs[i] + 1
                    elif pairs[i] != test_batch_size:
                        zero_sample = None
                    elif zeros_data:
                        continue

                    # 如果还有标签为1的样本，随机选择一个
                    if ones_data and pairs[i] < test_batch_size:
                        one_sample = random.choice(ones_data)
                        ones_data.remove(one_sample)  # 从列表中移除已选中的样本
                        batch_pairs[i].append(one_sample)
                        pairs[i] = pairs[i] + 1
                    elif pairs[i] != test_batch_size:
                        one_sample = None
                    elif ones_data:
                        continue
        else:
            # while zeros_data or ones_data:    ## 这种情况会导致循环停不下来
            while (len(zeros_data) + len(ones_data)) > (num_labels - test_batch_size * (num_batches - 1) - 2):
                # while (len(zeros_data)+len(ones_data)) >= num_batches-1:    ## 这种情况会导致之前有些列未填满，却也不满足"(len(zeros_data)+len(ones_data)) >= num_batches-1"，因此最后一组会把全部剩下的填补上导致pairs[num_batches-1]>test_batch_size
                for i in range(num_batches - 1):
                    # 如果还有标签为0的样本，随机选择一个
                    random.seed(seed)
                    if zeros_data and pairs[i] < test_batch_size:
                        zero_sample = random.choice(zeros_data)
                        zeros_data.remove(zero_sample)  # 从列表中移除已选中的样本
                        batch_pairs[i].append(zero_sample)
                        pairs[i] = pairs[i] + 1
                    elif pairs[i] != test_batch_size:
                        zero_sample = None
                    elif zeros_data:
                        continue

                    # 如果还有标签为1的样本，随机选择一个
                    if ones_data and pairs[i] < test_batch_size:
                        one_sample = random.choice(ones_data)
                        ones_data.remove(one_sample)  # 从列表中移除已选中的样本
                        batch_pairs[i].append(one_sample)
                        pairs[i] = pairs[i] + 1
                    elif pairs[i] != test_batch_size:
                        one_sample = None
                    elif ones_data:
                        continue
                # # 将选中的样本加入到当前批次
                # if zeros_data or ones_data:
                #     batch_pairs[i].append(zero_sample)
                #     batch_pairs[i].append(one_sample)
                # elif zeros_data:
                #     batch_pairs[i].append(zero_sample)
                # elif ones_data:
                #     batch_pairs[i].append(one_sample)

            # 将剩余样本加入到最后一组中
            if zeros_data and pairs[num_batches - 1] == 2:
                batch_pairs[num_batches - 1].extend(zeros_data)
                batch_pairs[num_batches - 1].extend(ones_data)
                # print("pairs[num_batches-1] == 2")
            elif ones_data and pairs[num_batches - 1] == 2:
                batch_pairs[num_batches - 1].extend(ones_data)
                batch_pairs[num_batches - 1].extend(zeros_data)
                # print("pairs[num_batches-1] == 2")

        # print(batch_pairs,len(batch_pairs))
        # flat_batches = [data for batch in batches for data in batch]
        for i in range(num_batches):
            batches.extend(batch_pairs[i])
        # print(batches,len(batches))
        # assert False

        # ## 查看是否正确按照标签进行分配
        # print()
        # data_y = [data.y for data in batches]
        # print(data_y)
        # assert False

        # Use DataLoader on the list of Data objects
        train_dataloader = DataLoader(train_data_list, batch_size=args.batch_size, shuffle=True)
        # test_dataloader = DataLoader(test_data_list, batch_size=args.batch_size, shuffle=False)
        # test_dataloader = DataLoader(test_data_list, batch_size=10, shuffle=True)
        test_dataloader = DataLoader(batches, batch_size=test_batch_size, shuffle=False)
        val_dataloader = DataLoader(val_data_list, batch_size=args.batch_size, shuffle=True)

        print(f'{time() - start_time:.2f} sec elapsed for data loading')
        return train_dataloader, test_dataloader, val_dataloader



    # print(test_data_list)    ## 原先的test_data_list形式[Data(), Data(), ...]
    ## 按照label存数据
    num_labels = len(test_data_list)
    # print(test_data_list)    ## [Data(x=[97, 8], edge_index=[2, 70], edge_attr=[70, 3], y=[1, 1]), Data(x=[97, 8], edge_index=[2, 100], edge_attr=[100, 3], y=[1, 1]), ...
    num_zeros = sum(1 for data in test_data_list if data.y == 0)
    num_ones = num_labels - num_zeros
    # print("num_zeros  num_ones",num_zeros,num_ones)    ## 71 81
    # if num_labels % 11 == 0:    ## 确保每一组至少一个为0和一个为1的标签
    #     test_batch_size = 11
    #     num_batches = num_labels // test_batch_size
    #     max_batch_size = test_batch_size
    if num_labels % 10 == 0:    ## 确保每一组至少一个为0和一个为1的标签
        test_batch_size = 10
        num_batches = num_labels // test_batch_size
        max_batch_size = test_batch_size
    elif num_labels % 9 == 0:
        test_batch_size = 9
        num_batches = num_labels // test_batch_size
        max_batch_size = test_batch_size
    elif num_labels % 8 == 0:
        test_batch_size = 8
        num_batches = num_labels // test_batch_size
        max_batch_size = test_batch_size
    elif num_labels % 7 == 0:
        test_batch_size = 7
        num_batches = num_labels // test_batch_size
        max_batch_size = test_batch_size
    else:    ## 10*9*8*7=5040
        if num_labels % 10 >= 2:
            test_batch_size = 10
            num_batches = num_labels // test_batch_size + 1
        elif num_labels % 9 >= 2:
            test_batch_size = 9
            num_batches = num_labels // test_batch_size + 1
        elif num_labels % 8 >= 2:
            test_batch_size = 8
            num_batches = num_labels // test_batch_size + 1
        elif num_labels % 7 >= 2:
            test_batch_size = 7
            num_batches = num_labels // test_batch_size + 1
        # num_batches =15   ## 将数据集分为15组,15组下test_batch_size不确定
        # test_batch_size = num_labels // num_batches
        # if num_labels % num_batches == 0:
        #     max_batch_size = test_batch_size
        # else:
        #     max_batch_size = test_batch_size + num_labels % num_batches

    # 将标签为0和1的样本分别存储起来
    zeros_data = [data for data in test_data_list if data.y == 0]
    ones_data = [data for data in test_data_list if data.y == 1]

    # 初始化批次列表
    batches = []
    batch_pairs = [[] for _ in range(num_batches)]
    pairs = [0 for _ in range(num_batches)]    ## counting
    seed = 42
    # 为每个批次随机选择一个标签为0和一个标签为1的样本
    for i in range(num_batches):
        # 如果还有标签为0的样本，随机选择一个
        random.seed(seed)
        if zeros_data:
            zero_sample = random.choice(zeros_data)
            zeros_data.remove(zero_sample)  # 从列表中移除已选中的样本
            batch_pairs[i].append(zero_sample)
            pairs[i] = pairs[i] + 1
        else:
            zero_sample = None

        # 如果还有标签为1的样本，随机选择一个
        if ones_data:
            one_sample = random.choice(ones_data)
            ones_data.remove(one_sample)  # 从列表中移除已选中的样本
            batch_pairs[i].append(one_sample)
            pairs[i] = pairs[i] + 1
        else:
            one_sample = None

    # for i in range(num_batches):
    #     data_y = [data.y for data in batch_pairs[i]]
    #     print(data_y)
    # assert False
        # # 将当前批次的样本对添加到 batch_pairs 中
        # batch_pairs[i].append(zero_sample)
        # batch_pairs[i].append(one_sample)
        # # 将选中的样本加入到当前批次
        # batch_pairs[i].append([zero_sample, one_sample])

    # print(num_batches)    ## 19
    # 向每个批次中添加样本，直到达到最大批次大小或所有样本都被使用   (如果没添加到)
    if num_labels % test_batch_size == 0:
        while zeros_data or ones_data:
            for i in range(num_batches):
                # 如果还有标签为0的样本，随机选择一个
                random.seed(seed)
                if zeros_data and pairs[i] < test_batch_size:
                    zero_sample = random.choice(zeros_data)
                    zeros_data.remove(zero_sample)  # 从列表中移除已选中的样本
                    batch_pairs[i].append(zero_sample)
                    pairs[i] = pairs[i]+1
                elif pairs[i]!=test_batch_size:
                    zero_sample = None
                elif zeros_data:
                    continue

                # 如果还有标签为1的样本，随机选择一个
                if ones_data and pairs[i] < test_batch_size:
                    one_sample = random.choice(ones_data)
                    ones_data.remove(one_sample)  # 从列表中移除已选中的样本
                    batch_pairs[i].append(one_sample)
                    pairs[i] = pairs[i] + 1
                elif pairs[i] != test_batch_size:
                    one_sample = None
                elif ones_data:
                    continue
    else:
        # while zeros_data or ones_data:    ## 这种情况会导致循环停不下来
        while (len(zeros_data)+len(ones_data)) > num_labels-test_batch_size*(num_batches-1)-2:
        # while (len(zeros_data)+len(ones_data)) >= num_batches-1:    ## 这种情况会导致之前有些列未填满，却也不满足"(len(zeros_data)+len(ones_data)) >= num_batches-1"，因此最后一组会把全部剩下的填补上导致pairs[num_batches-1]>test_batch_size
            for i in range(num_batches-1):
                # 如果还有标签为0的样本，随机选择一个
                random.seed(seed)
                if zeros_data and pairs[i] < test_batch_size:
                    zero_sample = random.choice(zeros_data)
                    zeros_data.remove(zero_sample)  # 从列表中移除已选中的样本
                    batch_pairs[i].append(zero_sample)
                    pairs[i] = pairs[i]+1
                elif pairs[i]!=test_batch_size:
                    zero_sample = None
                elif zeros_data:
                    continue

                # 如果还有标签为1的样本，随机选择一个
                if ones_data and pairs[i] < test_batch_size:
                    one_sample = random.choice(ones_data)
                    ones_data.remove(one_sample)  # 从列表中移除已选中的样本
                    batch_pairs[i].append(one_sample)
                    pairs[i] = pairs[i] + 1
                elif pairs[i] != test_batch_size:
                    one_sample = None
                elif ones_data:
                    continue
            # # 将选中的样本加入到当前批次
            # if zeros_data or ones_data:
            #     batch_pairs[i].append(zero_sample)
            #     batch_pairs[i].append(one_sample)
            # elif zeros_data:
            #     batch_pairs[i].append(zero_sample)
            # elif ones_data:
            #     batch_pairs[i].append(one_sample)

        # 将剩余样本加入到最后一组中
        if zeros_data and pairs[num_batches-1] == 2:
            batch_pairs[num_batches-1].extend(zeros_data)
            batch_pairs[num_batches - 1].extend(ones_data)
            # print("pairs[num_batches-1] == 2")
        elif ones_data and pairs[num_batches-1] == 2:
            batch_pairs[num_batches-1].extend(ones_data)
            batch_pairs[num_batches - 1].extend(zeros_data)
            # print("pairs[num_batches-1] == 2")

    # print(batch_pairs,len(batch_pairs))
    # flat_batches = [data for batch in batches for data in batch]
    for i in range(num_batches):
        batches.extend(batch_pairs[i])
    # print(batches,len(batches))
    # assert False

    # ## 查看是否正确按照标签进行分配
    # print()
    # data_y = [data.y for data in batches]
    # print(data_y)
    # assert False

    # Use DataLoader on the list of Data objects
    train_dataloader = DataLoader(train_data_list, batch_size=args.batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_data_list, batch_size=args.batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_data_list, batch_size=10, shuffle=True)
    test_dataloader = DataLoader(batches, batch_size=test_batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data_list, batch_size=args.batch_size, shuffle=True)

    print(f'{time() - start_time:.2f} sec elapsed for data loading')
    return train_dataloader, test_dataloader, val_dataloader

def process_molecule(x, adj, edge_attr, y):
    # Ensure x is a float tensor
    x_tensor = x.clone().detach().long() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)
    # Ensure y is a float tensor and convert to numpy array first if necessary
    # print("y:", y)
    if isinstance(y, torch.Tensor):
        y_tensor = y.clone().detach().float().reshape(1,-1)
    elif isinstance(y, (list, np.ndarray)):
        y_tensor = torch.from_numpy(np.array(y, dtype=np.float32)).float().reshape(1,-1)
    else:
        raise ValueError("y should be a numpy array, list, or torch tensor with shape [num_graph, 15]")

    # print("y_tensor:", y_tensor)
    #
    # with open('output_y.txt', 'a') as file:
    #     # file_samples.write(f"--------------------------------------------------------------\n")
    #     file.write(f"y:{y}\n")
    #     file.write(f"y_tensor:{y_tensor}\n\n")    ## y:[1]   y_tensor:tensor([[1.]])
    #
    #     file.write(f"--------------------------------------------------------------\n")
    # file.close()

    # Convert adjacency matrix to edge_index
    adj_tensor = adj.clone().detach() if isinstance(adj, torch.Tensor) else torch.tensor(adj, dtype=torch.float32)
    edge_index = torch.nonzero(adj_tensor, as_tuple=False).t().contiguous()
    if edge_index.nelement() == 0:
        edge_index = edge_index.view(2, 0)  # Ensure it's 2 x 0 for empty cases

    edge_attr_tensor = edge_attr.clone().detach() if isinstance(edge_attr, torch.Tensor) else torch.tensor(edge_attr,
                                                                                                           dtype=torch.float32)

    return Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr_tensor, y=y_tensor)

def create_data_list(mol_dataset):
    data_list = []
    for i in range(len(mol_dataset)):
        x, adj, edge_attr, y = mol_dataset[i]
        # print(y)  ##ogbg-molfreesolv使用expt一列
        data = process_molecule(x, adj, edge_attr, y)
        data_list.append(data)

    return data_list
