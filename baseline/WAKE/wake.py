import itertools

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.utils.data as Data
import copy

from baseline.WAKE.model import Encoder, reconstruct_Decoder, end2end_Decoder


class WAKE():
    def __init__(self, seed=None, beta=0.3, batch_size=64, n_epochs_1=14, n_epochs_2=10, n_epochs_3=4, lr=0.0002,
                 p_lambda=10, encoder_num_layers=4, decoder_num_layers=2, enc_hidden_dim=64, dec_hidden_dim=64):
        '''
        :param beta: Control the ratio of labeled anomalies to unlabeled samples
        :param batch_size:
        :param n_epochs_1: epoch of pre-training stage
        :param n_epochs_2: epoch of end-to-end  optimization stage
        :param n_epochs_3: epoch of fine-tuning stage
        :param p_lambda: a hyper-parameter to balance the contributions of two parts to the joint loss function
        :param lr: learning rate
        :param b1: adam: decay of first order momentum of gradient
        :param b2: adam: decay of first order momentum of gradient
        :param seed: value of Pytorch random seed
        :param enc_hidden_dim: hidden dimensions of BGRU layers in the encoder of feature encoder
        :param encoder_num_layers: number of BGRU layers in the encoder of feature encoder
        :param decoder_num_layers: number of GRU layers in the decoder of feature encoder
        :param dec_hidden_dim: hidden dimensions of GRU layers in the decoder of feature encoder
        '''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = 'WAKE'
        self.seed = seed
        self.beta = beta
        self.batch_size = batch_size
        self.n_epochs_1 = n_epochs_1
        self.n_epochs_2 = n_epochs_2
        self.n_epochs_3 = n_epochs_3
        self.lr = lr
        self.p_lambda = p_lambda
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        if type(self.seed) is int:
            torch.manual_seed(self.seed)

    def train_phase1(self, clean_dataloader, attribute_dims):
        '''
        :param clean_dataloader: only contains clean dataset
        :param attribute_dims:  Number of attribute values per attribute : list
        :return: encoder,decoder
        '''
        encoder = Encoder(attribute_dims, self.enc_hidden_dim, self.encoder_num_layers, self.dec_hidden_dim)
        decoder = reconstruct_Decoder(attribute_dims, self.enc_hidden_dim, self.decoder_num_layers, self.dec_hidden_dim)

        encoder.to(self.device)
        decoder.to(self.device)

        optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=self.lr)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(self.n_epochs_1 / 2), gamma=0.1)

        print("*" * 10 + "training_1" + "*" * 10)
        for epoch in range(int(self.n_epochs_1)):
            train_loss = 0.0
            train_num = 0
            for i, Xs in enumerate(tqdm(clean_dataloader)):
                mask = Xs[-1]
                Xs = Xs[:-1]
                mask = mask.to(self.device)
                for k, X in enumerate(Xs):
                    Xs[k] = X.to(self.device)

                s, enc_output = encoder(Xs)
                reconstruct_X = decoder(Xs, s, enc_output, mask)

                optimizer.zero_grad()

                loss = 0.0
                for ij in range(len(attribute_dims)):
                    # --------------
                    # 除了每一个属性的起始字符之外,其他重建误差
                    # ---------------
                    # pred=reconstruct_X[ij][:,1:,:].flatten(0,-2)
                    pred = torch.softmax(reconstruct_X[ij][:, 1:, :], dim=2).flatten(0, -2)
                    true = Xs[ij][:, 1:].flatten()
                    corr_pred = pred.gather(1, true.view(-1, 1)).flatten().to(self.device).reshape(-1,
                                                                                                   reconstruct_X[
                                                                                                       0].shape[
                                                                                                       1] - 1)

                    cross_entropys = -torch.log(corr_pred)
                    loss += cross_entropys.masked_select((~mask[:, 1:])).mean()

                train_loss += loss.item() * Xs[0].shape[0]
                train_num += Xs[0].shape[0]
                loss.backward()
                optimizer.step()
            ## 计算一个epoch在训练集上的损失和精度
            train_loss_epoch = train_loss / train_num
            print(f"[Epoch {epoch + 1:{len(str(self.n_epochs_1))}}/{self.n_epochs_1}] "
                  f"[loss: {train_loss_epoch:3f}]")
            scheduler.step()

        return encoder, decoder

    def train_phase2(self, dataloader, attribute_dims, max_len, encoder, decoder):
        '''
        :param dataloader: 平衡类别标签后的
        :param attribute_dims:  Number of attribute values per attribute : list
        :param max_len: max length of traces
        :param encoder: encoder of feature encoder
        :param decoder: decoder of feature encoder
        :return: end2end_decoder
        '''

        end2end_decoder = end2end_Decoder(len(attribute_dims) * self.dec_hidden_dim + len(attribute_dims) * (max_len))
        end2end_decoder.to(self.device)
        optimizer = torch.optim.Adam(
            itertools.chain(encoder.parameters(), end2end_decoder.parameters(), decoder.parameters()), lr=self.lr)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(self.n_epochs_2 / 2), gamma=0.1)
        print("*" * 10 + "training_2" + "*" * 10)
        for epoch in range(int(self.n_epochs_2)):
            train_loss = 0.0
            train_num = 0
            for i, Xs in enumerate(tqdm(dataloader)):
                labels = Xs[-1]
                mask = Xs[-2]
                Xs = Xs[:-2]
                for k, X in enumerate(Xs):
                    Xs[k] = X.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)
                s, enc_output = encoder(Xs)
                reconstruct_X = decoder(Xs, s, enc_output, mask)
                temp1 = []
                temp2 = []
                temp3 = []
                e = torch.zeros(len(labels)).to(self.device)
                for ij in range(len(attribute_dims)):
                    pred = torch.softmax(reconstruct_X[ij][:, 1:, :], dim=2).flatten(0, -2)
                    true = Xs[ij][:, 1:].flatten()
                    # probs.append(pred.gather(1,true.view(-1, 1)).flatten().to(device).reshape((-1,reconstruct_X[0].shape[1]-1)))
                    corr_pred = pred.gather(1, true.view(-1, 1)).flatten().to(self.device).reshape(-1,
                                                                                                   reconstruct_X[
                                                                                                       0].shape[
                                                                                                       1] - 1) * (
                                    ~mask[:, 1:])
                    corr_pred[corr_pred == 0] = 1
                    cross_entropys = -torch.log(corr_pred)

                    cross_entropy_max = cross_entropys.max(1).values.unsqueeze(1)  ##最大的损失
                    corr_pred_min = corr_pred.min(1).values.unsqueeze(1)
                    e += cross_entropys.sum(1) / (~mask[:, 1:]).sum(1)
                    temp1.append(cross_entropys)
                    temp2.append(cross_entropy_max)
                    temp3.append(corr_pred_min)
                temp1 = torch.cat(temp1, 1)
                temp2 = torch.cat(temp2, 1)
                temp3 = torch.cat(temp3, 1)

                trace_level_abnormal_scores = end2end_decoder(torch.cat((torch.cat(s, 1), temp1, temp2), 1)).squeeze()

                optimizer.zero_grad()

                loss = torch.mean((1 - labels) * trace_level_abnormal_scores + labels * torch.pow(
                    torch.log(trace_level_abnormal_scores), 2) + self.p_lambda * (
                                          (1 - labels) * e + labels * temp3.min(1).values))

                train_loss += loss.item() * Xs[0].shape[0]
                train_num += Xs[0].shape[0]
                loss.backward()
                optimizer.step()
            ## 计算一个epoch在训练集上的损失和精度
            train_loss_epoch = train_loss / train_num
            print(f"[Epoch {epoch + 1:{len(str(self.n_epochs_2))}}/{int(self.n_epochs_2)}] "
                  f"[loss: {train_loss_epoch:3f}]")
            scheduler.step()

        return end2end_decoder

    def train_phase3(self, dataloader, attribute_dims, encoder, decoder):
        '''
        :param dataloader: original dataset
        :param attribute_dims:  Number of attribute values per attribute : list
        :param encoder: encoder of feature encoder
        :param decoder: decoder of feature encoder
        :return: reconstruct_encoder,reconstruct_decoder
        '''

        reconstruct_encoder = copy.deepcopy(encoder)
        reconstruct_decoder = copy.deepcopy(decoder)

        optimizer = torch.optim.Adam(
            itertools.chain(reconstruct_encoder.parameters(), reconstruct_decoder.parameters()), lr=self.lr)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(self.n_epochs_3 / 2), gamma=0.1)
        print("*" * 10 + "training_3" + "*" * 10)
        for epoch in range(int(self.n_epochs_3)):
            train_loss = 0.0
            train_num = 0
            for i, Xs in enumerate(tqdm(dataloader)):
                labels = Xs[-1]
                mask = Xs[-2]
                Xs = Xs[:-2]
                mask = mask.to(self.device)
                labels = labels.to(self.device)
                for k, X in enumerate(Xs):
                    Xs[k] = X.to(self.device)

                s, enc_output = reconstruct_encoder(Xs)
                reconstruct_X = reconstruct_decoder(Xs, s, enc_output, mask)

                optimizer.zero_grad()

                loss = 0.0
                temp = []
                for ij in range(len(attribute_dims)):
                    # pred=reconstruct_X[ij][:,1:,:].flatten(0,-2)
                    pred = torch.softmax(reconstruct_X[ij][:, 1:, :], dim=2).flatten(0, -2)
                    true = Xs[ij][:, 1:].flatten()

                    corr_pred = pred.gather(1, true.view(-1, 1)).flatten().to(self.device).reshape(-1,
                                                                                                   reconstruct_X[
                                                                                                       0].shape[
                                                                                                       1] - 1) * (
                                    ~mask[:, 1:])
                    corr_pred[corr_pred == 0] = 1
                    cross_entropys = -torch.log(corr_pred)

                    corr_pred_min = corr_pred.min(1).values.unsqueeze(1)  ##每一个轨迹当前属性的最小的概率
                    cross_entropy_loss = cross_entropys.sum(1) / (~mask[:, 1:]).sum(1)  ##每一个轨迹当前属性的交叉熵损失

                    # loss +=  ((1-labels) *cross_entropy_loss).mean()

                    loss += (1 - labels) * cross_entropy_loss
                    temp.append(corr_pred_min)
                temp = torch.cat(temp, 1)
                loss = (loss + 2 * (labels * temp.min(1).values)).mean()

                train_loss += loss.item() * Xs[0].shape[0]
                train_num += Xs[0].shape[0]
                loss.backward()
                optimizer.step()
            ## 计算一个epoch在训练集上的损失和精度
            train_loss_epoch = train_loss / train_num
            print(f"[Epoch {epoch + 1:{len(str(self.n_epochs_3))}}/{int(self.n_epochs_3)}] "
                  f"[loss: {train_loss_epoch:3f}]")
            scheduler.step()

        return reconstruct_encoder, reconstruct_decoder

    def fit(self, dataset):
        Xs_clean = []
        for i, dim in enumerate(dataset.attribute_dims):
            Xs_clean.append(torch.LongTensor(np.delete(dataset.features[i], dataset.labeled_indices, 0)))
        clean_mask = torch.BoolTensor(np.delete(dataset.mask, dataset.labeled_indices, 0))
        clean_dataset = Data.TensorDataset(*Xs_clean, clean_mask)
        clean_dataloader = DataLoader(clean_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                      pin_memory=True,
                                      drop_last=True)
        encoder, decoder = self.train_phase1(clean_dataloader, dataset.attribute_dims)

        anomalies_num = int(dataset.weak_labels.sum())
        repeat_times = int((len(dataset.weak_labels) * self.beta) / ((1 - self.beta) * anomalies_num))
        train_Xs = []
        for i, dim in enumerate(dataset.attribute_dims):
            train_Xs.append(torch.LongTensor(dataset.features[i]))
        train_labels = torch.LongTensor(dataset.weak_labels)
        for i in dataset.labeled_indices:
            for j, dim in enumerate(dataset.attribute_dims):
                train_Xs[j] = torch.cat((train_Xs[j], train_Xs[j][i].repeat((repeat_times, 1))))
        train_mask = torch.BoolTensor(dataset.mask)
        for i in dataset.labeled_indices:
            train_mask = torch.cat((train_mask, train_mask[i].repeat((repeat_times, 1))))
        train_labels = torch.cat((train_labels, torch.ones(len(dataset.labeled_indices) * repeat_times)))
        train_dataset = Data.TensorDataset(*train_Xs, train_mask, train_labels)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                      pin_memory=True, drop_last=True)

        end2end_decoder = self.train_phase2(train_dataloader, dataset.attribute_dims, dataset.max_len, encoder, decoder)

        Xs = []
        for i, dim in enumerate(dataset.attribute_dims):
            Xs.append(torch.LongTensor(dataset.features[i]))
        mask = torch.BoolTensor(dataset.mask)
        labels = torch.LongTensor(dataset.weak_labels)

        ori_dataset = Data.TensorDataset(*Xs, mask, labels)
        ori_dataloader = DataLoader(ori_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                    pin_memory=True,
                                    drop_last=True)
        reconstruct_encoder, reconstruct_decoder = self.train_phase3(ori_dataloader, dataset.attribute_dims, encoder,
                                                                     decoder)

        self.encoder = encoder
        self.decoder = decoder
        self.end2end_decoder = end2end_decoder
        self.reconstruct_encoder = reconstruct_encoder
        self.reconstruct_decoder = reconstruct_decoder

    def detect(self, dataset):

        Xs = []
        for i, dim in enumerate(dataset.attribute_dims):
            Xs.append(torch.LongTensor(dataset.features[i]))
        mask = torch.BoolTensor(dataset.mask)

        detect_dataset = Data.TensorDataset(*Xs, mask)

        detect_dataloader = DataLoader(detect_dataset, batch_size=self.batch_size,
                                       shuffle=False, num_workers=0, pin_memory=True)

        attr_Shape = (dataset.num_cases, dataset.max_len, dataset.num_attributes)
        self.encoder.eval()
        self.decoder.eval()
        self.end2end_decoder.eval()
        self.reconstruct_encoder.eval()
        self.reconstruct_decoder.eval()

        pos = 0
        with torch.no_grad():
            trace_level_abnormal_scores = []
            attr_level_abnormal_scores = np.zeros(attr_Shape)
            print("*" * 10 + "detecting" + "*" * 10)
            for Xs in tqdm(detect_dataloader):
                mask_c = Xs[-1]
                Xs = Xs[:-1]
                for k, X in enumerate(Xs):
                    Xs[k] = X.to(self.device)
                mask = mask_c.to(self.device)

                s, enc_output = self.encoder(Xs)
                temp_X = self.decoder(Xs, s, enc_output, mask)

                temp1 = []
                temp2 = []
                for ij in range(len(dataset.attribute_dims)):
                    pred = torch.softmax(temp_X[ij][:, 1:, :], dim=2).flatten(0, -2)
                    true = Xs[ij][:, 1:].flatten()
                    corr_pred = pred.gather(1, true.view(-1, 1)).flatten().to(self.device).reshape(-1,
                                                                                              temp_X[0].shape[
                                                                                                  1] - 1) * (
                                    ~mask[:, 1:])
                    corr_pred[corr_pred == 0] = 1
                    cross_entropys = -torch.log(corr_pred)
                    cross_entropy_max = cross_entropys.max(1).values.unsqueeze(1)  ##最大的损失
                    temp1.append(cross_entropys)
                    temp2.append(cross_entropy_max)

                temp1 = torch.cat(temp1, 1)
                temp2 = torch.cat(temp2, 1)

                trace_level_abnormal_score = self.end2end_decoder(torch.cat((torch.cat(s, 1), temp1, temp2), 1))

                trace_level_abnormal_scores.append(trace_level_abnormal_score.detach().cpu())

                s, enc_output = self.reconstruct_encoder(Xs)
                reconstruct_X = self.reconstruct_decoder(Xs, s, enc_output, mask)

                for attr_index in range(len(dataset.attribute_dims)):
                    reconstruct_X[attr_index] = reconstruct_X[attr_index]
                    reconstruct_X[attr_index] = torch.softmax(reconstruct_X[attr_index], dim=2)

                mask[:, 0] = True  # 第一个事件是我们添加的起始事件，屏蔽掉

                for attr_index in range(len(dataset.attribute_dims)):
                    truepos = Xs[attr_index].flatten()
                    p = reconstruct_X[attr_index].reshape((truepos.shape[0], -1)).gather(1,
                                                                                         truepos.view(-1, 1)).squeeze()
                    p_distribution = reconstruct_X[attr_index].reshape((truepos.shape[0], -1))

                    p_distribution = p_distribution + 1e-8  # 避免出现概率为0

                    attr_level_abnormal_scores[pos: pos + Xs[attr_index].shape[0], :, attr_index] = \
                        (torch.sigmoid(
                            (torch.sum(torch.log(p_distribution) * p_distribution, 1) - torch.log(p)).reshape(
                                Xs[attr_index].shape)) * (~mask)).detach().cpu()
                pos += Xs[attr_index].shape[0]

            event_level_abnormal_scores = attr_level_abnormal_scores.max((2))

            trace_level_abnormal_scores = torch.cat(trace_level_abnormal_scores, dim=0).flatten()

            return trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores