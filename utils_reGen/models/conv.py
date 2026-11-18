import torch
import torch.nn.functional as F
# Notes: Only works for ogbg dataset
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from torch_geometric.nn import MessagePassing, GATConv, SAGEConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.norm import GraphNorm, PairNorm, DiffGroupNorm, InstanceNorm, LayerNorm, GraphSizeNorm
from torch_geometric.utils import degree

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

nn_act = torch.nn.ReLU()  # ReLU activation
F_act = F.relu


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            nn_act,
            torch.nn.Linear(2 * emb_dim, emb_dim)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
        return out

    def message(self, x_j, edge_attr):
        if edge_attr is None:
            return F_act(x_j)
        return F_act(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm) + F_act(
            x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        if edge_attr is None:
            return norm.view(-1, 1) * F_act(x_j)
        return norm.view(-1, 1) * F_act(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, in_edge_dim, in_feat_dim, emb_dim, drop_ratio=0.5, JK="last", residual=False,
                 gnn_name='gin', norm_layer='batch_norm', **kwargs):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
            gnn_name (str): type of GNN ('gin', 'gcn', 'gat', 'sage')
            kwargs: additional arguments for specific GNN layers (e.g., heads for GAT)
        '''
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.norm_layer = norm_layer
        self.gnn_name = gnn_name

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # 编码器
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)
        self.uni_bond_encoder = Uni_BondEncoder(in_edge_dim, emb_dim)
        self.uni_atom_encoder = Uni_AtomEncoder(in_feat_dim, emb_dim)

        # 初始化卷积层
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_name == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_name == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            elif gnn_name == 'gat':
                heads = kwargs.get('heads', 4)  # 默认头数
                concat = kwargs.get('concat', True)
                self.convs.append(
                    GATConv(in_channels=emb_dim, out_channels=emb_dim // heads,
                            heads=heads, concat=concat, dropout=drop_ratio))
            elif gnn_name == 'sage':
                aggr = kwargs.get('aggr', 'mean')
                self.convs.append(SAGEConv(emb_dim, emb_dim, aggr=aggr))
            else:
                raise ValueError(f'Undefined GNN type called {gnn_name}')

            # 初始化归一化层
            norm = self.initialize_norm(norm_layer, emb_dim)
            self.batch_norms.append(norm)

        if norm_layer.split('_')[1] == 'size':
            self.graph_size_norm = GraphSizeNorm()

    def initialize_norm(self, norm_layer, emb_dim):
        norm_type = norm_layer.split('_')[0]
        if norm_type == 'batch':
            if norm_layer.split('_')[-1] == 'notrack':
                return torch.nn.BatchNorm1d(emb_dim, track_running_stats=False, affine=False)
            else:
                return torch.nn.BatchNorm1d(emb_dim)
        elif norm_type == 'instance':
            return InstanceNorm(emb_dim)
        elif norm_type == 'layer':
            return LayerNorm(emb_dim)
        elif norm_type == 'graph':
            return GraphNorm(emb_dim)
        elif norm_type == 'size':
            return GraphSizeNorm()
        elif norm_type == 'pair':
            return PairNorm(emb_dim)
        elif norm_type == 'group':
            return DiffGroupNorm(emb_dim, groups=4)
        else:
            raise ValueError(f'Undefined normalization layer called {norm_layer}')

    def forward(self, batched_data, encode_raw=True):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        # 计算输入节点嵌入
        if encode_raw:
            h_list = [self.uni_atom_encoder(x)]
            edge_attr_encoded = self.uni_bond_encoder(edge_attr)
        else:
            h_list = [x]
            edge_attr_encoded = edge_attr

        for layer in range(self.num_layer):
            if self.gnn_name == 'sage':
                # SAGEConv 不使用 edge_attr
                h = self.convs[layer](h_list[layer], edge_index)
            else:
                # GIN, GCN, GAT 使用 edge_attr
                h = self.convs[layer](h_list[layer], edge_index, edge_attr_encoded)

            # 处理归一化
            if self.norm_layer.split('_')[0] == 'batch':
                h = self.batch_norms[layer](h)
            else:
                h = self.batch_norms[layer](h, batch)

            if self.norm_layer.split('_')[1] == 'size':
                h = self.graph_size_norm(h, batch)

            # 激活函数和 dropout
            if layer == self.num_layer - 1:
                # 最后一层不使用激活函数
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F_act(h)
                h = F.dropout(h, self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

        # Jk-concat 的不同实现
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = sum(h_list)
        else:
            raise ValueError(f"Unsupported JK option: {self.JK}")

        return node_representation, h_list


class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, in_edge_dim, in_feat_dim, emb_dim, drop_ratio=0.5, JK="last", residual=False,
                 gnn_name='sage', norm_layer='batch_norm', **kwargs):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
            gnn_name (str): type of GNN ('gin', 'gcn', 'gat', 'sage')
            kwargs: additional arguments for specific GNN layers (e.g., heads for GAT)
        '''
        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.norm_layer = norm_layer
        self.gnn_name = gnn_name

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # 编码器
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)
        self.uni_bond_encoder = Uni_BondEncoder(in_edge_dim, emb_dim)
        self.uni_atom_encoder = Uni_AtomEncoder(in_feat_dim, emb_dim)

        # 初始化虚拟节点嵌入
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        # 初始化卷积层
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_name == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_name == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            elif gnn_name == 'gat':
                heads = kwargs.get('heads', 4)  # 默认头数
                concat = kwargs.get('concat', True)
                self.convs.append(
                    GATConv(in_channels=emb_dim, out_channels=emb_dim // heads, heads=heads,
                            concat=concat, dropout=drop_ratio))
            elif gnn_name == 'sage':
                aggr = kwargs.get('aggr', 'mean')
                self.convs.append(SAGEConv(emb_dim, emb_dim, aggr=aggr))
            else:
                raise ValueError(f'Undefined GNN type called {gnn_name}')

            # 初始化归一化层
            norm = self.initialize_norm(norm_layer, emb_dim)
            self.batch_norms.append(norm)

        if norm_layer.split('_')[1] == 'size':
            self.graph_size_norm = GraphSizeNorm()

        # 初始化用于更新虚拟节点的 MLP
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim), nn_act,
                torch.nn.Linear(2 * emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), nn_act))

    def initialize_norm(self, norm_layer, emb_dim):
        norm_type = norm_layer.split('_')[0]
        if norm_type == 'batch':
            if norm_layer.split('_')[-1] == 'notrack':
                return torch.nn.BatchNorm1d(emb_dim, track_running_stats=False, affine=False)
            else:
                return torch.nn.BatchNorm1d(emb_dim)
        elif norm_type == 'instance':
            return InstanceNorm(emb_dim)
        elif norm_type == 'layer':
            return LayerNorm(emb_dim)
        elif norm_type == 'graph':
            return GraphNorm(emb_dim)
        elif norm_type == 'size':
            return GraphSizeNorm()
        elif norm_type == 'pair':
            return PairNorm(emb_dim)
        elif norm_type == 'group':
            return DiffGroupNorm(emb_dim, groups=4)
        else:
            raise ValueError(f'Undefined normalization layer called {norm_layer}')

    def forward(self, batched_data, encode_raw=True):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        # 获取虚拟节点嵌入
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1, dtype=torch.long, device=edge_index.device)
        )

        # 计算输入节点嵌入
        if encode_raw:
            h_list = [self.uni_atom_encoder(x)]
            edge_attr_encoded = self.uni_bond_encoder(edge_attr)
        else:
            h_list = [x]
            edge_attr_encoded = edge_attr

        for layer in range(self.num_layer):
            # 将虚拟节点信息添加到图节点
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            # 图节点间的信息传递
            if self.gnn_name == 'sage':
                # SAGEConv 不使用 edge_attr
                h = self.convs[layer](h_list[layer], edge_index)
            else:
                # GIN, GCN, GAT 使用 edge_attr
                h = self.convs[layer](h_list[layer], edge_index, edge_attr_encoded)

            # 处理归一化
            if self.norm_layer.split('_')[0] == 'batch':
                h = self.batch_norms[layer](h)
            else:
                h = self.batch_norms[layer](h, batch)

            if self.norm_layer.split('_')[1] == 'size':
                h = self.graph_size_norm(h, batch)

            # 激活函数和 dropout
            if layer == self.num_layer - 1:
                # 最后一层不使用激活函数
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F_act(h)
                h = F.dropout(h, self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            # 更新虚拟节点嵌入
            if layer < self.num_layer - 1:
                # 从图节点向虚拟节点传递信息
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.drop_ratio,
                        training=self.training
                    )
                else:
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.drop_ratio,
                        training=self.training
                    )

        # Jk-concat 的不同实现
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = sum(h_list)
        else:
            raise ValueError(f"Unsupported JK option: {self.JK}")

        return node_representation, h_list


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        # print('full_atom_feature_dims', len(full_atom_feature_dims))
        # assert False
        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim, max_norm=1)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()
        # print('full_bond_feature_dims',len(full_bond_feature_dims))
        # assert False
        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim, max_norm=None)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = torch.zeros(edge_attr.size(0), self.bond_embedding_list[0].embedding_dim,
                                     device=edge_attr.device, dtype=torch.float32)
        # if edge_attr.is_cuda:
        #     print('Edge attribute in cuda')
        for i in range(edge_attr.shape[1]):
            # bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])
            indices = edge_attr[:, i].long()
            bond_embedding += self.bond_embedding_list[i](indices)

        return bond_embedding


class Uni_AtomEncoder(torch.nn.Module):

    def __init__(self, in_dim, emb_dim):
        super(Uni_AtomEncoder, self).__init__()

        self.atom_embedding = torch.nn.Linear(in_dim, emb_dim)
        torch.nn.init.xavier_uniform_(self.atom_embedding.weight.data)

    def forward(self, x):
        self.atom_embedding = self.atom_embedding.to(x.device)
        atom_embedding = self.atom_embedding(x.float())

        return atom_embedding


class Uni_BondEncoder(torch.nn.Module):

    def __init__(self, in_dim, emb_dim):
        super(Uni_BondEncoder, self).__init__()

        self.bond_embedding = torch.nn.Linear(in_dim, emb_dim)
        torch.nn.init.xavier_uniform_(self.bond_embedding.weight.data)

    def forward(self, edge_attr):
        self.bond_embedding = self.bond_embedding.to(edge_attr.device)
        bond_embedding = self.bond_embedding(edge_attr)

        return bond_embedding
