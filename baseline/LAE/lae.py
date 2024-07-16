import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from baseline.LAE.model import LSTMAE

import torch.utils.data as Data
from torch.utils.data import DataLoader
class LAE():
    def __init__(self, batch_size=16, n_epochs=100, lr=0.0001 ,b1=0.9 ,b2=0.999 ,seed=None, hidden_size = 64):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size=batch_size
        self.seed = seed
        self.n_epochs = n_epochs
        self.lr =lr
        self.b1 =b1
        self.b2 =b2
        self.hidden_size = hidden_size
        self.name = 'LAE'

        if type(self.seed) is int:
            torch.manual_seed(self.seed)


    def fit(self, dataset):
        X = dataset.flat_onehot_features
        X = torch.Tensor(X)
        # print(dataset.event_log)
        case_lens = torch.LongTensor(dataset.case_lens)
        tensorDataset = Data.TensorDataset(X, case_lens)

        dataloader = DataLoader(tensorDataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                                drop_last=True)

        attribute_dim_index = torch.LongTensor(
            [sum(dataset.attribute_dims[:i + 1]) for i in range(len(dataset.attribute_dims))])

        self.model = LSTMAE(int(dataset.flat_onehot_features.shape[-1]), self.hidden_size, num_layers=2, device=self.device)
        loss_func = nn.MSELoss()

        self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

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
                                       shuffle=False, num_workers=0, pin_memory=True)

        self.model.eval()

        predictions = []
        for X, case_len in tqdm(detect_dataloader):
            X = X.to(self.device)

            fake_X = self.model(X)

            fake_X = fake_X.flatten(1, 2)

            predictions.append(fake_X.detach().cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)

        # Calculate error
        errors = np.power(dataset.flat_onehot_features_2d - predictions, 2)
        errors = errors * np.expand_dims(~dataset.mask, 2).repeat(dataset.attribute_dims.sum(), 2).reshape(
            dataset.mask.shape[0], -1)

        trace_level_abnormal_scores = errors.sum(1) / (dataset.case_lens * dataset.attribute_dims.sum())

        # Split the errors according to the events
        split_event = np.cumsum(np.tile(dataset.attribute_dims.sum(), [dataset.max_len]), dtype=int)[:-1]
        errors_event = np.split(errors, split_event, axis=1)
        errors_event = np.array([np.mean(a, axis=1) if len(a) > 0 else 0.0 for a in errors_event])
        event_level_abnormal_scores = errors_event.T

        # Split the errors according to the attribute dims
        split = np.cumsum(np.tile(dataset.attribute_dims, [dataset.max_len]), dtype=int)[:-1]
        errors_attr = np.split(errors, split, axis=1)
        errors_attr = np.array([np.mean(a, axis=1) if len(a) > 0 else 0.0 for a in errors_attr])

        # Init anomaly scores array
        attr_level_abnormal_scores = np.zeros(dataset.binary_targets.shape)

        for i in range(len(dataset.attribute_dims)):
            error = errors_attr[i::len(dataset.attribute_dims)]
            attr_level_abnormal_scores[:, :, i] = error.T

        return trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores