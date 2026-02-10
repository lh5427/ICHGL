import torch
from torch_geometric.typing import OptTensor
from torch import Tensor
from torch_scatter import scatter_add, scatter_softmax
from torch_geometric.nn.conv import MessagePassing

class GraphConvLayer(MessagePassing):
    """
    Lightweight Graph Convolution Layer which supports gcn and tide normalization.
    """
    def __init__(self, in_channels, out_channels, norm_type):
        super(GraphConvLayer, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm_type
    
    # LightGCN normalization
    def gcn_norm(self, edge_index, num_nodes, weight=None):

        if weight == None:
            edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
        else:
            edge_weight = weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    # 修改后的tide_norm函数：
    def tide_norm(self, edge_index, x, target_emb, num_nodes):
        key, query = edge_index[0], edge_index[1]  # row, col

        # GAT形式的连接操作：拼接源节点目标嵌入和目标节点特征
        attention_input = torch.cat([target_emb[key], x[query]], dim=-1)

        # 创建注意力权重和偏置，并确保它们在正确的设备上
        device = attention_input.device  # 获取输入张量的设备
        attention_weight = torch.nn.Parameter(torch.Tensor(attention_input.size(-1), 1)).to(device)
        attention_bias = torch.nn.Parameter(torch.Tensor(1)).to(device)

        # 初始化参数
        torch.nn.init.xavier_uniform_(attention_weight)
        torch.nn.init.zeros_(attention_bias)

        # 计算注意力分数：线性变换 + 偏置
        attention_score = torch.matmul(attention_input, attention_weight) + attention_bias

        edge_weight = scatter_softmax(attention_score, key, dim=0, dim_size=num_nodes)
        return edge_index, edge_weight



    def forward(self, x, edge_index, target_emb=None, aux_target_emb=None):
        num_nodes = x.size(0)

        if self.norm_type == 'tide':
            # 使用 aux_target_emb 作为 TIDE 归一化中的目标嵌入
            tide_target_emb = aux_target_emb if aux_target_emb is not None else target_emb
            edge_index, edge_weight = self.tide_norm(edge_index, x, tide_target_emb, num_nodes)
        elif self.norm_type == 'gcn':
            edge_index, edge_weight = self.gcn_norm(edge_index, num_nodes)
        else:
            raise ValueError('Invalid normalization type')

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j