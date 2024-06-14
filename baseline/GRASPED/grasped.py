import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

import torch.utils.data as Data
from torch.utils.data import DataLoader

from baseline.GRASPED.GRU_AE import GRU_AE


class GRASPED():
    def __init__(self, batch_size=64, n_epochs=20 ,lr=0.0002 ,b1=0.5 ,b2=0.999 ,seed=None ,enc_hidden_dim = 64 , encoder_num_layers = 4,decoder_num_layers=2, dec_hidden_dim = 64):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size=batch_size
        self.seed = seed
        self.n_epochs = n_epochs
        self.lr =lr
        self.b1 =b1
        self.b2 =b2
        self.enc_hidden_dim = enc_hidden_dim
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.dec_hidden_dim = dec_hidden_dim
        self.name = 'GRASPED'

        if type(self.seed) is int:
            torch.manual_seed(self.seed)


    def fit(self, dataset):
        attribute_dims=dataset.attribute_dims
        Xs = []
        for i, dim in enumerate(attribute_dims):
            Xs.append(torch.LongTensor(dataset.features[i]))
        mask = torch.BoolTensor(dataset.mask)
        Dataset = Data.TensorDataset(*Xs, mask)

        dataloader = DataLoader(Dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                      pin_memory=True, drop_last=True)

        self.model = GRU_AE(dataset.attribute_dims, self.enc_hidden_dim, self.encoder_num_layers, self.decoder_num_layers, self.dec_hidden_dim)

        self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(self.n_epochs / 2), gamma=0.1)

        print("*" * 10 + "training" + "*" * 10)
        for epoch in range(int(self.n_epochs)):
            train_loss = 0.0
            train_num = 0
            for i, Xs in enumerate(tqdm(dataloader)):
                mask = Xs[-1]
                Xs = Xs[:-1]
                mask = mask.to(self.device)
                for k, X in enumerate(Xs):
                    Xs[k] = X.to(self.device)

                fake_X = self.model(Xs,mask)

                optimizer.zero_grad()

                loss = 0.0
                for ij in range(len(attribute_dims)):
                    # --------------
                    # 除了每一个属性的起始字符之外,其他重建误差
                    # ---------------
                    pred = torch.softmax(fake_X[ij][:, 1:, :], dim=2).flatten(0, -2)
                    true = Xs[ij][:, 1:].flatten()

                    corr_pred = pred.gather(1, true.view(-1, 1)).flatten().to(self.device).reshape(-1,
                                                                                              fake_X[0].shape[1] - 1)

                    cross_entropys = -torch.log(corr_pred)
                    loss += cross_entropys.masked_select((~mask[:, 1:])).mean()

                    # loss+=loss_func(pred,true)

                train_loss += (loss.item() * Xs[0].shape[0]) /len(dataset.attribute_dims)
                train_num += Xs[0].shape[0]
                loss.backward()
                optimizer.step()
            ## 计算一个epoch在训练集上的损失和精度
            train_loss_epoch = train_loss / train_num
            print(f"[Epoch {epoch + 1:{len(str(self.n_epochs))}}/{self.n_epochs}] "
                  f"[loss: {train_loss_epoch:3f}]")
            scheduler.step()

        return self

    def detect(self, dataset):
        Xs = []
        attribute_dims = dataset.attribute_dims
        for i, dim in enumerate(dataset.attribute_dims):
            Xs.append(torch.LongTensor(dataset.features[i]))
        mask = torch.BoolTensor(dataset.mask)
        tensorDataset = Data.TensorDataset(*Xs,mask)

        dataloader = DataLoader(tensorDataset, batch_size=self.batch_size,
                                       shuffle=False, num_workers=0, pin_memory=True)
        self.model.eval()
        with torch.no_grad():
            print("*" * 10 + "detecting" + "*" * 10)
            final_res = []
            for Xs in tqdm(dataloader):
                mask = Xs[-1]
                Xs = Xs[:-1]

                mask = mask.to(self.device)
                for k, tempX in enumerate(Xs):
                    Xs[k] = tempX.to(self.device)

                fake_X = self.model(Xs,mask)

                for attr_index in range(len(attribute_dims)):
                    fake_X[attr_index] = torch.softmax(fake_X[attr_index], dim=2)

                this_res = []
                for attr_index in range(len(attribute_dims)):
                    temp = fake_X[attr_index]
                    index = Xs[attr_index].unsqueeze(2)
                    probs = temp.gather(2, index)
                    temp[(temp <= probs)] = 0
                    res = temp.sum(2)
                    res = res * (~mask)
                    this_res.append(res)

                final_res.append(torch.stack(this_res, 2))

            attr_level_abnormal_scores = np.array(torch.cat(final_res, 0).detach().cpu())
            trace_level_abnormal_scores = attr_level_abnormal_scores.max((1, 2))
            event_level_abnormal_scores = attr_level_abnormal_scores.max((2))
            return trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores
