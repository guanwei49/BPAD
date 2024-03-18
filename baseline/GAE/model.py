import math
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import Module

from torch_geometric.nn import NNConv


class callable_my(Module):
    def __init__(self, out_dim, device):
        super().__init__()
        self.device=device
        self.out_dim=out_dim
    def forward(self,edge_attr):
        return torch.ones(self.out_dim).to(self.device)


class GAEModel(nn.Module):
    def __init__(self, node_dim,hidden_dim, device ,with_edge_attr=False, edge_dim=None):
        super().__init__()
        # 定义 NNConv 层
        # 设置 aggr='mean' 表示使用边特征进行信息聚合
        self.with_edge_attr=with_edge_attr
        if with_edge_attr:
            self.ecc = NNConv(node_dim, hidden_dim, nn=nn.Sequential(
                nn.Linear(edge_dim, node_dim*hidden_dim), # [-1, num_edge_features] to shape [-1, in_channels * out_channels]
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(p=0.3)
            ), aggr='mean')

            self.ecc2 = NNConv(hidden_dim, hidden_dim, nn=nn.Sequential(
                nn.Linear(edge_dim, hidden_dim*hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(p=0.3)
            ), aggr='mean')
        else:
            self.ecc = NNConv(node_dim, hidden_dim, nn=callable_my( node_dim * hidden_dim,device ), aggr='mean')

            self.ecc2 = NNConv(hidden_dim, hidden_dim, nn=callable_my( hidden_dim * hidden_dim,device ), aggr='mean')
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_dim * 2, 1)

    def forward(self, graph, node_nums):
        # 对节点进行信息聚合
        # 注意这里需要传入边特征 edge_attr
        if self.with_edge_attr:
            x = self.ecc(graph.x, graph.edge_index, graph.edge_attr)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.ecc2(x, graph.edge_index, graph.edge_attr)
        else:
            x = self.ecc(graph.x, graph.edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.ecc2(x, graph.edge_index)

        # 应用 ReLU 激活函数
        x = F.relu(x)

        batch_ms = []

        h = torch.split(x, tuple(node_nums), dim=0)
        for batch, node_num in enumerate(node_nums):
            ms = []
            for i in range(node_num):
                temp = h[batch][i].repeat((node_num, 1))
                ms.append(torch.cat((temp, h[batch]),1))
            ms = torch.cat(ms, 0)
            batch_ms.append(ms)
        batch_ms = torch.cat(batch_ms, 0)
        batch_ms = self.linear(batch_ms)
        batch_ms = torch.sigmoid(batch_ms)
        batch_ms = batch_ms.squeeze()

        batch_ms = torch.split(batch_ms, tuple(node_nums * node_nums), dim=0)

        return batch_ms


