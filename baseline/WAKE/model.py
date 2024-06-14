import torch.nn.functional as F
import torch
from torch import nn

class Encoder_sig(nn.Module):
    def __init__(self, input_size, enc_hidden_dim, num_layers, dec_hid_dim):
        super().__init__()
        self.input_size = input_size
        self.enc_hiddenc_dim = enc_hidden_dim
        self.num_layers = num_layers
        self.dec_hid_dim = dec_hid_dim
        self.embedding = nn.Embedding(self.input_size, self.enc_hiddenc_dim)
        self.gru = nn.GRU(input_size = enc_hidden_dim,
                          hidden_size = enc_hidden_dim,
                          num_layers = num_layers,
                          dropout=0.3, bidirectional=True)


        self.fc = nn.Sequential(
            nn.BatchNorm1d(enc_hidden_dim * 2),
            nn.Linear(enc_hidden_dim * 2, dec_hid_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3)
        )

    def forward(self, init_input):
        self.gru.flatten_parameters()
        init_input= self.embedding(init_input)
        enc_output, hidden = self.gru(init_input,  None)
        # 将正向最后时刻的输出与反向最后时刻的输出进行拼接，得到的维度应该是[batch,enc_hiddenc_dim*2]
        # hidden[-1,:,:] 反向最后时刻的输出, hidden[-2,:,:] 正向最后时刻的输出
        h_m = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) # 在第一个维度进行拼接
        s0 = self.fc(h_m)
        return enc_output,s0  # enc_output:[seq_len,batch,enc_hiddenc_dim*2] s0:[batch,dec_hiddenc_dim]


class end2end_Decoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim , int(input_dim/3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d( int(input_dim/3)),
            nn.Linear(int(input_dim/3), int(input_dim/9)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(int(input_dim / 9), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.MLP(x)

class end2end_Decoder_ablation(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim , 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.MLP(x)

class Attention(nn.Module):
    '''
    加性注意力
    '''
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)  # 输出的维度是任意的
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)  # 将输出维度置为1

    def forward(self, s, enc_output ,mask):
        # s = [batch_size, dec_hidden_dim]
        # enc_output = [seq_len*num_attrs, batch_size, enc_hid_dim * 2]
        # mask = [batch_size,seq_len]

        all_len = enc_output.shape[0]

        # repeat decoder hidden state seq_len times
        # s = [seq_len, batch_size, dec_hid_dim]
        s = s.repeat(all_len, 1,1)  # [batch_size, dec_hid_dim]=>[seq_len*num_attrs, batch_size, dec_hid_dim]

        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

        attention = self.v(energy).squeeze(
            2).transpose(0, 1).unsqueeze(1)  # [seq_len*num_attrs, batch_size, dec_hid_dim]=>[seq_len*num_attrs，batch_size, 1] => [batch_size ,1, seq_len*num_attrs ]

        mask = mask.unsqueeze(1)
        num_attr = int(attention.shape[2] / mask.shape[2])
        mask = mask.repeat((1, 1, num_attr))

        attention[mask] = float('-inf')

        attention_probs=F.softmax(attention, dim=-1)  # [batch_size, 1 , seq_len*num_attrs]

        enc_output = enc_output.transpose(0, 1)

        # # c = [1, batch_size, enc_hid_dim * 2]
        c = torch.bmm(attention_probs, enc_output).transpose(0, 1)

        return c,attention_probs


class Decoder_sig(nn.Module):
    def __init__(self, vocab_size, enc_hid_dim, dec_hid_dim,num_layers,output_dim):
        super().__init__()
        self.num_layers=num_layers
        self.vocab_size = vocab_size
        self.attention =  Attention(enc_hid_dim, dec_hid_dim)
        self.embedding = nn.Embedding(vocab_size, enc_hid_dim)
        self.rnn = nn.GRU(enc_hid_dim * 2+ enc_hid_dim, dec_hid_dim,num_layers = num_layers,dropout=0.3)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + enc_hid_dim , output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, dec_input, s, enc_output,mask):
        # dec_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [seq_len*num_attrs, batch_size, enc_hid_dim * 2]
        # mask = [batch_size,seq_len]
        self.rnn.flatten_parameters()

        dec_input = dec_input.unsqueeze(1) # dec_input = [batch_size, 1]
        dec_input =self.embedding(dec_input)  # dec_input = [batch_size, 1] => [batch_size, 1,enc_hid_dim]

        dropout_dec_input = self.dropout(dec_input).transpose(0, 1)  #  [batch_size, 1,enc_hid_dim]=>[1,batch_size,enc_hid_dim]

        # c = [1, batch_size, enc_hid_dim * 2]
        c,attention_probs = self.attention(s, enc_output, mask)

        rnn_input = torch.cat((dropout_dec_input, c), dim = 2) # rnn_input = [1, batch_size, (enc_hid_dim * 2)+ enc_hid_dim]

        dec_output, dec_hidden = self.rnn(rnn_input, s.repeat( self.num_layers,1,1))

        dec_output = dec_output.squeeze(0) # dec_output:[ batch_size, dec_hid_dim]

        c = c.squeeze(0)  # c:[batch_size, enc_hid_dim * 2]

        dropout_dec_input=dropout_dec_input.squeeze(0)  # dec_input:[batch_size, enc_hid_dim]

        pred = self.fc_out(torch.cat((dec_output, c, dropout_dec_input), dim = 1))# pred = [batch_size, output_dim]

        return pred, dec_hidden[-1].squeeze(0),attention_probs

class Encoder(nn.Module):
    def __init__(self, attribute_dims, enc_hidden_dim, encoder_num_layers, dec_hidden_dim):
        super().__init__()
        encoders = []
        self.attribute_dims = attribute_dims
        for i, dim in enumerate(attribute_dims):
            encoders.append(Encoder_sig(int(dim + 1), enc_hidden_dim, encoder_num_layers, dec_hidden_dim))
        self.encoders = nn.ModuleList(encoders)

    def forward(self, Xs):
        '''
        :param Xs:是多个属性，每一个属性作为一个X ：[batch_size, seq_len]
        :return:
        '''
        s = []  # 解码层GRU初始隐藏表示
        enc_output = None
        for i, dim in enumerate(self.attribute_dims):
            X = Xs[i].transpose(0, 1)

            enc_output_, s_ = self.encoders[i](X)

            if enc_output is None:
                enc_output = enc_output_
            else:
                enc_output = torch.cat((enc_output, enc_output_), dim=0)
            # enc_output = [seq_len*num_attrs, batch_size, enc_hid_dim * 2]
            s.append(s_)


        return s,enc_output




class reconstruct_Decoder(nn.Module):
    def __init__(self, attribute_dims,enc_hidden_dim,decoder_num_layers,dec_hidden_dim):
        super().__init__()
        decoders=[]
        self.attribute_dims=attribute_dims
        for i, dim in enumerate(attribute_dims):
            decoders.append(Decoder_sig(int(attribute_dims[0] + 1), enc_hidden_dim, dec_hidden_dim, decoder_num_layers, int(dim + 1)))

        self.decoders = nn.ModuleList(decoders)


    def forward(self, Xs,s,enc_output,mask):
        '''
        :param enc_output = [seq_len*num_attrs, batch_size, enc_hid_dim * 2]
        :param Xs:是多个属性，每一个属性作为一个X ：[batch_size, seq_len]
        :return:
        '''
        trg_len = None
        reconstruct_outputs = []     #概率分布 probability map
        for i, dim in enumerate(self.attribute_dims):
            X = Xs[i].transpose(0, 1)
            batch_size = X.shape[1]
            trg_len = X.shape[0]
            output_dim = int(dim)+1
            # 存储所有时刻的输出
            reconstruct_outputs.append(torch.zeros(trg_len, batch_size, output_dim).to(X.device))  # 存储decoder的所有输出

            reconstruct_outputs[i][0, :, X[0, 0]] = 1     #存储预测结果


        for i, dim in enumerate(self.attribute_dims):
            if i == 0:
                X = Xs[i].transpose(0, 1)
                s0 = s[i]
                dec_input = X[0, :]  # target的第一列，即全是起始字符 teacher_forcing

                for t in range(1, trg_len):
                    dec_output, s0, _ = self.decoders[i](dec_input, s0, enc_output,mask)

                    # 存储每个时刻的输出
                    reconstruct_outputs[i][t] = dec_output

                    dec_input = X[t]  # teacher_forcing

                reconstruct_outputs[i] = reconstruct_outputs[i].transpose(0, 1)
            else:
                X = Xs[0].transpose(0, 1)  # teacher_forcing 只看activity
                s0 = s[i]

                for t in range(1, trg_len):
                    dec_input = X[t]  # teacher_forcing  只看activity    (因为属性值与活动名称有关)

                    dec_output, s0, _ = self.decoders[i](dec_input, s0, enc_output, mask)

                    # 存储每个时刻的输出
                    reconstruct_outputs[i][t] = dec_output

                reconstruct_outputs[i] = reconstruct_outputs[i].transpose(0, 1)


        return reconstruct_outputs

