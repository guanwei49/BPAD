import numpy as np
import torch
from torch import nn, optim
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import random
from baseline.GAMA.GAT_AE import GAT_AE


class GAMA():
    def __init__(self, seed=None ,n_epochs=20 ,batch_size= 64, lr=0.0005 ,b1=0.5 ,b2=0.999 ,hidden_dim = 64, GAT_heads = 4, decoder_num_layers = 2,TF_styles:str='FAP'):
        if TF_styles not in ['AN', 'PAV', 'FAP']:
            raise Exception('"TF_styles" must be a value in ["AN","PAV", "FAP"]')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.n_epochs = n_epochs
        self.lr =lr
        self.b1 =b1
        self.b2 =b2
        self.batch_size=batch_size
        self.hidden_dim =hidden_dim
        self.GAT_heads =GAT_heads
        self.decoder_num_layers =decoder_num_layers
        self.TF_styles=TF_styles
        self.name = 'GAMA'
        if type(self.seed) is int:
            torch.manual_seed(self.seed)


    def fit(self, dataset):
        self.model = GAT_AE(dataset.attribute_dims, dataset.max_len, self.hidden_dim, self.GAT_heads, self.decoder_num_layers, self.TF_styles)
        loss_func = nn.CrossEntropyLoss()

        self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=0.85
        )

        Xs = []
        for i, dim in enumerate(dataset.attribute_dims):
            Xs.append(torch.LongTensor(dataset.features[i]))

        print("*" * 10 + "training" + "*" * 10)
        for epoch in range(int(self.n_epochs)):
            train_loss = 0.0
            train_num = 0
            # 自定义的dataloader
            indexes = [i for i in range(len(dataset))]  # 打乱顺序
            random.shuffle(indexes)
            for bathc_i in tqdm(range(self.batch_size, len(indexes)+1, self.batch_size)):
                this_batch_indexes = indexes[bathc_i - self.batch_size:bathc_i]
                nodes_list = [dataset.node_xs[i] for i in this_batch_indexes]
                edge_indexs_list = [dataset.edge_indexs[i] for i in this_batch_indexes]
                Xs_list = []
                graph_batch_list = []
                for i in range(len(dataset.attribute_dims)):
                    Xs_list.append(Xs[i][this_batch_indexes].to(self.device))
                    graph_batch = Batch.from_data_list([Data(x=nodes_list[b][i], edge_index=edge_indexs_list[b])
                                                        for b in range(len(nodes_list))])
                    graph_batch_list.append(graph_batch.to(self.device))
                mask = torch.tensor(dataset.mask[this_batch_indexes]).to(self.device)

                attr_reconstruction_outputs = self.model(graph_batch_list, Xs_list, mask, len(this_batch_indexes))

                optimizer.zero_grad()

                loss = 0.0
                mask[:, 0] = True  # 除了每一个属性的起始字符之外,其他重建误差
                for i in range(len(dataset.attribute_dims)):
                    # --------------
                    # 除了每一个属性的起始字符之外,其他重建误差
                    # ---------------
                    pred = attr_reconstruction_outputs[i][~mask]
                    true = Xs_list[i][~mask]
                    loss += loss_func(pred, true)

                train_loss += loss.item()/len(dataset.attribute_dims)
                train_num += 1
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
            final_res = []
            attribute_dims = dataset.attribute_dims

            Xs = []
            for i, dim in enumerate(dataset.attribute_dims):
                Xs.append(torch.LongTensor(dataset.features[i]))

            print("*" * 10 + "detecting" + "*" * 10)

            pre = 0

            for bathc_i in tqdm(range(self.batch_size, len(dataset) + self.batch_size, self.batch_size)):
                if bathc_i <= len(dataset):
                    this_batch_indexes = list(range(pre, bathc_i))
                else:
                    this_batch_indexes = list(range(pre, len(dataset)))

                nodes_list = [dataset.node_xs[i] for i in this_batch_indexes]
                edge_indexs_list = [dataset.edge_indexs[i] for i in this_batch_indexes]
                Xs_list = []
                graph_batch_list = []
                for i in range(len(dataset.attribute_dims)):
                    Xs_list.append(Xs[i][this_batch_indexes].to(self.device))
                    graph_batch = Batch.from_data_list([Data(x=nodes_list[b][i], edge_index=edge_indexs_list[b])
                                                        for b in range(len(nodes_list))])
                    graph_batch_list.append(graph_batch.to(self.device))
                mask = torch.tensor(dataset.mask[this_batch_indexes]).to(self.device)

                attr_reconstruction_outputs =   self.model(graph_batch_list, Xs_list, mask, len(this_batch_indexes))

                for attr_index in range(len(attribute_dims)):
                    attr_reconstruction_outputs[attr_index] = torch.softmax(attr_reconstruction_outputs[attr_index],
                                                                            dim=2)

                this_res = []
                for attr_index in range(len(attribute_dims)):
                    # 取比实际出现的属性值大的其他属性值的概率之和
                    temp = attr_reconstruction_outputs[attr_index]
                    index = Xs_list[attr_index].unsqueeze(2)
                    probs = temp.gather(2, index)
                    temp[(temp <= probs)] = 0
                    res = temp.sum(2)
                    res = res * (~mask)
                    this_res.append(res)

                final_res.append(torch.stack(this_res, 2))

                pre = bathc_i

            attr_level_abnormal_scores = np.array(torch.cat(final_res, 0).detach().cpu())
            trace_level_abnormal_scores = attr_level_abnormal_scores.max((1, 2))
            event_level_abnormal_scores = attr_level_abnormal_scores.max((2))
            return trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores


