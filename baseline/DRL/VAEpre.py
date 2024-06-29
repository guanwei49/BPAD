import numpy as np
import torch.utils.data as Data
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm




class VAEModel(nn.Module):
    # x --> fc1 --> relu --> fc2 --> relu --> fc3 --> z --> fc3  --> relu --> fc4 --> relu -->fc5 --> x'
    def __init__(self, input_size, device):
        '''
        layer1: dim of hidden layer1
        layer2: dim of hidden layer2
        '''
        super(VAEModel, self).__init__()

        self.input_size = input_size
        self.device = device

        layer1 = int(input_size / 2)
        layer2 = int(input_size / 4)
        layer3 = 5
        self.fc1 = nn.Linear(input_size, layer1)
        self.fc2 = nn.Linear(layer1, layer2)  # encode
        self.fc31 = nn.Linear(layer2, layer3)  # encode
        self.fc32 = nn.Linear(layer2, layer3)  # encode
        self.fc4 = nn.Linear(layer3, layer2)  # decode
        self.fc5 = nn.Linear(layer2, layer1)  # decode
        self.fc6 = nn.Linear(layer1, input_size)  # decode

        self.relu = nn.ReLU()

    def encode(self, x):
        # x --> fc1 --> relu --> fc2 --> relu --> fc31
        # x --> fc1 --> relu --> fc2 --> relu --> fc32
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps * std + mu

    def decode(self, z, x):
        # z --> fc4 --> relu --> fc5  --> relu --> fc6
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.fc6(h5).view(x.size())

    def forward(self, x):
        # flatten input and pass to encode
        # mu= mean
        # logvar = log variational
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparametrize(mu, logvar)
        return self.decode(z, x), mu, logvar, z


def loss_function(recon_x, x, mu, logvar, avai_mask):
    MSE = F.mse_loss(recon_x * avai_mask, x * avai_mask, reduction='sum')
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    loss = MSE + KLD
    return loss

def PrewithVAE(dataset, device, batch_size = 32, n_epochs = 500, lr=0.001):
    activitySeq = dataset.onehot_features[0]  # adaptor： only remains control flow
    activitySeq = activitySeq.reshape((activitySeq.shape[0], np.prod(activitySeq.shape[1:])))


    activitySeq_unlabeled = torch.Tensor(np.delete(activitySeq, dataset.labeled_indices, 0))
    case_lens_unlabeled = torch.LongTensor(np.delete(dataset.case_lens, dataset.labeled_indices, 0))
    tensorDataset = Data.TensorDataset(activitySeq_unlabeled, case_lens_unlabeled)
    dataloader = DataLoader(tensorDataset, batch_size = batch_size, shuffle=True, num_workers=0, pin_memory=True,
                            drop_last=True)

    activitySeq = torch.Tensor(activitySeq)
    case_lens = torch.LongTensor(dataset.case_lens)
    tensorDataset = Data.TensorDataset(activitySeq, case_lens)
    detect_dataloader = DataLoader(tensorDataset, batch_size=batch_size,
                                   shuffle=False, num_workers=0, pin_memory=True)

    model = VAEModel(activitySeq_unlabeled.shape[1], device=device)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    print("*" * 10 + "training VAE" + "*" * 10)
    for epoch in range(int(n_epochs)):
        train_loss = 0.0
        train_num = 0
        for X, case_len in tqdm(dataloader):
            X = X.to(device)
            mask = torch.zeros(X.shape).to(device)

            fake_X, mu, logvar, z = model(X)

            optimizer.zero_grad()

            for p, mylen in enumerate(dataset.attribute_dims[0] * case_len):
                mask[p, :mylen] = 1

            loss = loss_function(fake_X, X, mu, logvar, mask)

            train_loss += (loss.item() * X[0].shape[0])
            train_num += X[0].shape[0]
            loss.backward()
            optimizer.step()

        ## 计算一个epoch在训练集上的损失和精度
        train_loss_epoch = train_loss / train_num
        print(f"[Epoch {epoch + 1:{len(str(n_epochs))}}/{n_epochs}] "
              f"[loss: {train_loss_epoch:3f}]")


    z_l = []
    ri_l = []

    print("*" * 10 + "Preprocessing data with VAE" + "*" * 10)
    for X, case_len in tqdm(detect_dataloader):
        X = X.to(device)
        mask = torch.zeros(X.shape).to(device)

        fake_X, _, _, z = model(X)

        for p, mylen in enumerate(dataset.attribute_dims[0] * case_len):
            mask[p, :mylen] = 1

        ri = torch.norm(fake_X * mask - X, p=2, dim=1)

        z_l.append(z)
        ri_l.append(ri)

    z_l = torch.concatenate(z_l)
    ri_l = torch.concatenate(ri_l)

    min_val = torch.min(ri_l)
    max_val = torch.max(ri_l)

    # 进行 min-max 归一化
    ri_l = (ri_l - min_val) / (max_val - min_val)

    ## z_l:(num_sample, 5)    ri_l: (num_sample, 1)
    return z_l.detach().cpu().numpy(), ri_l.detach().cpu().numpy()
