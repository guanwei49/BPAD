import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import random
from baseline.GAE.model import GAEModel
from torch_geometric.data import Data, Batch

class GAE():
    def __init__(self, seed=None ,batch_size=64, n_epochs=20, lr=0.0005 ,b1=0.5 ,b2=0.999, hidden_dim=64):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.n_epochs = n_epochs
        self.lr =lr
        self.b1 =b1
        self.b2 =b2
        self.batch_size = batch_size
        self.hidden_dim =hidden_dim
        self.name='GAE'



    def fit(self, dataset):
        if type(self.seed) is int:
            torch.manual_seed(self.seed)
        if dataset.trace_graphs_GAE[0].edge_attr is not None:
            self.model = GAEModel(int(dataset.attribute_dims[0]), self.hidden_dim, self.device, True, len(dataset.trace_graphs_GAE[0].edge_attr[0]))
        else:
            self.model = GAEModel(int(dataset.attribute_dims[0]), self.hidden_dim,self.device)
        loss_func = nn.BCELoss()

        self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=0.85
        )

        print("*" * 10 + "training" + "*" * 10)
        for epoch in range(int(self.n_epochs)):
            train_num = 0
            train_loss = 0

            indexes = [i for i in range(len(dataset))]  # 打乱顺序
            random.shuffle(indexes)

            for bathc_i in tqdm(range(self.batch_size, len(indexes) + 1, self.batch_size)):
                this_batch_indexes = indexes[bathc_i - self.batch_size:bathc_i]
                batch_=[dataset.trace_graphs_GAE[index] for index in this_batch_indexes ]
                graph_batch = Batch.from_data_list(batch_)
                node_nums = np.array([d.x.shape[0] for d in batch_])
                graph_batch = graph_batch.to(self.device)

                batch_ms = self.model(graph_batch, node_nums)

                optimizer.zero_grad()
                loss = 0
                for ith, ms in enumerate(batch_ms):
                    this_graph =  batch_[ith]

                    edge_index_T = this_graph.edge_index.T
                    target = torch.zeros(len(ms)).to(self.device)
                    for edge in edge_index_T:
                        target[len(this_graph.x) * edge[0] + edge[1]] = 1
                    loss += loss_func(ms, target)

                train_loss += loss.item()
                train_num += len(this_batch_indexes)
                loss.backward()
                optimizer.step()
            ## 计算一个epoch在训练集上的损失和精度
            train_loss_epoch = train_loss / train_num
            print(f"[Epoch {epoch + 1:{len(str(self.n_epochs))}}/{self.n_epochs}] "
                  f"[loss: {train_loss_epoch:3f}]")
            scheduler.step()
        return self

    def detect(self, dataset):
        self.model.eval()
        with torch.no_grad():
            ##useless
            attr_Shape = (dataset.num_cases, dataset.max_len, dataset.num_attributes)
            attr_level_abnormal_scores = np.zeros(attr_Shape)

            loss_func = nn.BCELoss()

            trace_level_abnormal_scores = []

            print("*" * 10 + "detecting" + "*" * 10)
            pre = 0

            for bathc_i in tqdm(range(self.batch_size, len(dataset) + self.batch_size, self.batch_size)):
                if bathc_i <= len(dataset):
                    this_batch_indexes = list(range(pre, bathc_i))
                else:
                    this_batch_indexes = list(range(pre, len(dataset)))

                batch_ = [dataset.trace_graphs_GAE[index] for index in this_batch_indexes]
                graph_batch = Batch.from_data_list(batch_)
                node_nums = np.array([d.x.shape[0] for d in batch_])
                graph_batch = graph_batch.to(self.device)

                batch_ms = self.model(graph_batch, node_nums)

                for ith, ms in enumerate(batch_ms):
                    this_graph = batch_[ith]

                    edge_index_T = this_graph.edge_index.T
                    target = torch.zeros(len(ms)).to(self.device)
                    for edge in edge_index_T:
                        target[len(this_graph.x) * edge[0] + edge[1]] = 1
                    loss = loss_func(ms, target)
                    trace_level_abnormal_scores.append(loss.detach().cpu())
                pre = bathc_i

        trace_level_abnormal_scores = np.array(trace_level_abnormal_scores)
        ##useless
        event_level_abnormal_scores = attr_level_abnormal_scores.max((2))

        return trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores

