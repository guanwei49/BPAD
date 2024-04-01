import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from baseline.LAE.model import LSTMAE

import torch.utils.data as Data
from torch.utils.data import DataLoader
class LAE():
    def __init__(self, batch_size=64, n_epochs=20 ,lr=0.0002 ,b1=0.5 ,b2=0.999 ,seed=None, hidden_size = 64):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size=batch_size
        self.seed = seed
        self.n_epochs = n_epochs
        self.lr =lr
        self.b1 =b1
        self.b2 =b2
        self.hidden_size = hidden_size
        self.name = 'LAE'


    def fit(self, dataset):
        X = dataset.flat_onehot_features
        X = torch.Tensor(X)
        # print(dataset.event_log)
        case_lens = torch.LongTensor(dataset.case_lens)
        tensorDataset = Data.TensorDataset(X, case_lens)

        dataloader = DataLoader(tensorDataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                                drop_last=True)

        if type(self.seed) is int:
            torch.manual_seed(self.seed)
        attribute_dim_index = torch.LongTensor(
            [sum(dataset.attribute_dims[:i + 1]) for i in range(len(dataset.attribute_dims))])

        if type(self.seed) is int:
            torch.manual_seed(self.seed)

        self.model = LSTMAE(int(dataset.flat_onehot_features.shape[-1]), self.hidden_size, num_layers=2, isCuda=(self.device.type == 'cuda'))
        loss_func = nn.MSELoss()

        self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        print("*" * 10 + "training" + "*" * 10)
        for epoch in range(int(self.n_epochs)):
            train_loss = 0.0
            train_num = 0
            for X, case_len in tqdm(dataloader):
                X = X.to(self.device)

                fake_X = self.model(X)

                optimizer.zero_grad()

                # only consider active part
                temp = torch.LongTensor(np.ones(fake_X.shape))

                for k in range(temp.shape[0]):
                    temp[k, case_len[k] * attribute_dim_index[-1]:] = 0

                temp = torch.LongTensor(temp).to(self.device)
                fake_X = fake_X * temp
                X = X * temp

                loss = loss_func(fake_X, X)

                train_loss += (loss.item() * X.shape[0])
                train_num += X.shape[0]
                loss.backward()
                optimizer.step()

            ## 计算一个epoch在训练集上的损失和精度
            train_loss_epoch = train_loss / train_num
            print(f"[Epoch {epoch + 1:{len(str(self.n_epochs))}}/{self.n_epochs}] "
                  f"[loss: {train_loss_epoch:3f}]")
            scheduler.step()

        return self

    def detect(self, dataset):
        X = dataset.flat_onehot_features
        X = torch.Tensor(X)
        # print(dataset.event_log)
        case_lens = torch.LongTensor(dataset.case_lens)
        tensorDataset = Data.TensorDataset(X, case_lens)

        detect_dataloader = DataLoader(tensorDataset, batch_size=self.batch_size,
                                       shuffle=False, num_workers=4, pin_memory=True)

        attr_Shape = (dataset.num_cases, dataset.max_len, dataset.num_attributes)

        self.model.eval()
        attribute_dim_index = torch.LongTensor(
            [sum(dataset.attribute_dims[:i + 1]) for i in range(len(dataset.attribute_dims))])
        criterion = nn.MSELoss()

        index = -1
        trace_level_abnormal_scores = []
        attr_level_abnormal_scores = np.zeros(attr_Shape)
        event_level_abnormal_scores = np.zeros((attr_Shape[0], attr_Shape[1]))
        for X, case_len in tqdm(detect_dataloader):
            X = X.to(self.device)

            fake_X = self.model(X)

            for trace_i in range(X.shape[0]):
                one_X = X[trace_i]
                one_fake_X = fake_X[trace_i]
                index += 1
                one_fake_X = one_fake_X.flatten()
                one_X = one_X.flatten()

                loss_X = criterion(one_fake_X[:case_len[trace_i] * attribute_dim_index[-1]],
                                   one_X[:case_len[trace_i] * attribute_dim_index[-1]])

                anomaly_score = loss_X

                trace_level_abnormal_scores.append(anomaly_score.cpu().item())

                lastend = 0
                for i in range(case_len[trace_i]):
                    for j in range(len(attribute_dim_index)):
                        end = int(i * attribute_dim_index[-1] + attribute_dim_index[j])
                        attr_level_abnormal_scores[index][i][j] = criterion(one_fake_X[lastend:end], one_X[lastend:end])
                        lastend = end

                lastend = 0
                for i in range(case_len[trace_i]):
                    end = int((i + 1) * attribute_dim_index[-1])
                    event_score = criterion(one_fake_X[lastend:end], one_X[lastend:end])
                    lastend = end
                    event_level_abnormal_scores[index][i] = event_score

        trace_level_abnormal_scores = np.array(trace_level_abnormal_scores)

        return trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores