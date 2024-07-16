import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader
from sklearn.svm import OneClassSVM
from baseline.VAEOCSVM.model import VAEModel


class VAEOCSVM():
    def __init__(self, batch_size=64, n_epochs=100 ,lr=0.0001 ,b1=0.5 ,b2=0.999 ,seed=None ,hidden_size = 64):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size=batch_size
        self.seed = seed
        self.n_epochs = n_epochs
        self.lr =lr
        self.b1 =b1
        self.b2 =b2
        self.hidden_size =hidden_size
        self.name = 'VAE-OCSVM'

        if type(self.seed) is int:
            torch.manual_seed(self.seed)

    def loss_function(self, recon_x, x, mu, logvar, avai_mask):
        MSE = F.mse_loss(recon_x * avai_mask, x * avai_mask, size_average=False)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        loss = MSE + KLD
        return loss

    def fit(self, dataset):
        activity = dataset.onehot_features[0]  # adaptor： only remains control flow
        X = activity.reshape((activity.shape[0], np.prod(activity.shape[1:])))
        X = torch.Tensor(X)
        case_lens = torch.LongTensor(dataset.case_lens)
        tensorDataset = Data.TensorDataset(X, case_lens)

        dataloader = DataLoader(tensorDataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                                drop_last=True)

        self.model =  VAEModel(int(X.shape[-1]), int(self.hidden_size * 3), self.hidden_size, device=self.device)

        self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))


        print("*" * 10 + "training" + "*" * 10)
        for epoch in range(int(self.n_epochs)):
            train_loss = 0.0
            train_num = 0
            for X, case_len in tqdm(dataloader):
                X = X.to(self.device)
                mask = torch.zeros(X.shape).to(self.device)

                fake_X, mu, logvar,_ = self.model(X)

                optimizer.zero_grad()

                for p, mylen in enumerate(dataset.attribute_dims.sum()*case_len):
                    mask[p,:mylen]=1

                loss = self.loss_function(fake_X, X, mu, logvar,mask)

                train_loss += (loss.item() * X[0].shape[0])
                train_num += X[0].shape[0]
                loss.backward()
                optimizer.step()

            ## 计算一个epoch在训练集上的损失和精度
            train_loss_epoch = train_loss / train_num
            print(f"[Epoch {epoch + 1:{len(str(self.n_epochs))}}/{self.n_epochs}] "
                  f"[loss: {train_loss_epoch:3f}]")

        return self

    def detect(self, dataset):
        activity = dataset.flat_onehot_features[:, :, :dataset.attribute_dims[0]]  #adaptor： only remains control flow
        X = activity.reshape((activity.shape[0], np.prod(activity.shape[1:])))
        X = torch.Tensor(X)

        tensorDataset = Data.TensorDataset(X)
        detect_dataloader = DataLoader(tensorDataset, batch_size=self.batch_size,
                                       shuffle=False, num_workers=0, pin_memory=True)

        self.model.eval()
        zs=[]
        for X in tqdm(detect_dataloader):
            X = X[0].to(self.device)

            fake_X, _, _, z = self.model(X)

            zs.append(z)

        zs=torch.cat(zs,dim=0)

        zs=zs.detach().cpu()
        zs=zs.numpy()

        clf = OneClassSVM(gamma='auto', nu=0.5) #-1 for outliers and 1 for inliers.

        clf.fit(zs)

        scores = -clf.score_samples(zs)

        trace_level_abnormal_scores = (scores-scores.min())/(scores.max()-scores.min())


        return trace_level_abnormal_scores, None, None

