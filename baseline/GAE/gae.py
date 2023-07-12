import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from baseline.GAE.model import GAEModel


class GAE():
    def __init__(self, seed=None ,n_epochs=2 ,lr=0.0002 ,b1=0.5 ,b2=0.999 ,hidden_dim=64):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.n_epochs = n_epochs
        self.lr =lr
        self.b1 =b1
        self.b2 =b2
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

        print("*" * 10 + "training" + "*" * 10)
        for epoch in range(int(self.n_epochs)):
            train_num = 0
            train_loss = 0
            for index, graph in enumerate(tqdm(dataset.trace_graphs_GAE)):
                graph = graph.to(self.device)
                ms =  self.model(graph)

                optimizer.zero_grad()

                edge_index_T = graph.edge_index.T
                target = torch.zeros(len(ms)).to(self.device)
                for edge in edge_index_T:
                    target[len(graph.x) * edge[0] + edge[1]] = 1
                loss = loss_func(ms, target)
                train_loss += loss.item()
                train_num += 1
                loss.backward()
                optimizer.step()
            ## 计算一个epoch在训练集上的损失和精度
            train_loss_epoch = train_loss / train_num
            print(f"[Epoch {epoch + 1:{len(str(self.n_epochs))}}/{self.n_epochs}] "
                  f"[loss: {train_loss_epoch:3f}]")

        return self

    def detect(self, dataset):
        self.model.eval()

        attr_Shape = (dataset.num_cases, dataset.max_len, dataset.num_attributes)
        attr_level_abnormal_scores = np.zeros(attr_Shape)

        loss_func = nn.BCELoss()

        trace_level_abnormal_scores = []

        for index, graph in enumerate(tqdm(dataset.trace_graphs_GAE)):
            graph = graph.to(self.device)
            ms = self.model(graph)

            edge_index_T = graph.edge_index.T
            target = torch.zeros(len(ms)).to(self.device)
            for edge in edge_index_T:
                target[len(graph.x) * edge[0] + edge[1]] = 1
            loss = loss_func(ms, target)
            trace_level_abnormal_scores.append(loss.detach().cpu())

        trace_level_abnormal_scores = np.array(trace_level_abnormal_scores)

        event_level_abnormal_scores = attr_level_abnormal_scores.max((2))

        return trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores

