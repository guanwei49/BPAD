import math
import torch.nn.functional as F
import torch
from torch import nn
from torch_geometric.nn import GATConv


class PositionalEncoder(nn.Module):
    def __init__(self, input_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i

        pe = torch.zeros(max_seq_len, input_dim)
        for pos in range(max_seq_len):
            for i in range(0, input_dim, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / input_dim)))
                if i+1 < input_dim:
                    pe[pos, i + 1] = \
                        math.cos(pos / (10000 ** ((2 * (i + 1)) / input_dim)))
        self.register_buffer('pe', pe)

    def forward(self, x,batch_size):
        # make embeddings relatively larger
        x = x * math.sqrt(self.input_dim)
        # add constant to embedding
        batch_pe= self.pe.repeat((batch_size,1,1))
        x = x + batch_pe.reshape((-1,batch_pe.shape[2]))
        return self.dropout(x)


class GAT_Encoder(nn.Module):
    def __init__(self, input_dim, enc_hidden_dim, in_head , max_seq_len):
        super().__init__()
        # self.embed = nn.Linear(input_dim, input_dim)

        self.pe = PositionalEncoder(input_dim,max_seq_len)

        self.conv1 = GATConv(in_channels=input_dim,
                             out_channels=enc_hidden_dim,
                             heads=in_head,
                             dropout=0.4)
        self.conv2 = GATConv(in_channels=enc_hidden_dim * in_head,
                             out_channels=enc_hidden_dim,
                             concat=False,
                             heads=1,
                             dropout=0.4)

    def forward(self, data,batch_size):
        x, edge_index = data.x, data.edge_index
        # x = self.embed(x)
        x = self.pe(x,batch_size)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index) # x:[batch_size*seq_len , enc_hidden_dim]
        x = x.reshape(batch_size,-1,x.shape[1]) # x:[batch_size, seq_len , enc_hidden_dim]

        return x


# class Attention(nn.Module):
#     '''
#     加性注意力
#     '''
#     def __init__(self, enc_hid_dim, dec_hid_dim):
#         super().__init__()
#
#         self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim, bias=False)  # 输出的维度是任意的
#         self.v = nn.Linear(dec_hid_dim, 1, bias=False)  # 将输出维度置为1
#
#     def forward(self, s, enc_output):
#         # s = [batch_size, dec_hidden_dim]
#         # enc_output = [seq_len, batch_size, enc_hid_dim * 2]
#
#         seq_len = enc_output.shape[0]
#
#         # repeat decoder hidden state seq_len times
#         # s = [seq_len, batch_size, dec_hid_dim]
#         s = s.repeat(seq_len, 1,1)  # [batch_size, dec_hid_dim]=>[seq_len, batch_size, dec_hid_dim]
#
#         energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))
#
#         attention = self.v(energy).squeeze(
#             2)  # [seq_len, batch_size, dec_hid_dim]=>[seq_len，batch_size, 1] => [seq_len, batch_size]
#
#         attention_probs=F.softmax(attention, dim=0).transpose(0, 1).unsqueeze(1)  # [batch_size, 1 , seq_len]
#
#         enc_output = enc_output.transpose(0, 1)
#
#         # # c = [1, batch_size, enc_hid_dim * 2]
#         c = torch.bmm(attention_probs, enc_output).transpose(0, 1)
#
#         return c,attention_probs


class Attention(nn.Module):
    '''
    点积注意力
    '''
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.hidden=enc_hid_dim
        self.query = nn.Linear(dec_hid_dim, self.hidden)
        self.key = nn.Linear(enc_hid_dim, self.hidden)

    def forward(self, s, enc_output,mask):
        # s = [batch_size, dec_hidden_dim]
        # enc_output = [self.max_seq_len*len(self.attribute_dims), batch_size, enc_hid_dim ]
        s = s.unsqueeze(0) # [batch_size, dec_hid_dim]=>[1, batch_size, dec_hid_dim]
        s=s.transpose(0, 1) # [1, batch_size, dec_hid_dim] => [batch_size,1 dec_hid_dim]
        q=self.query(s) # [batch_size,1 , self.hidden]
        enc_output=enc_output.transpose(0, 1)  # [batch_size, , enc_hid_dim ]
        k=self.key(enc_output) # [batch_size,self.max_seq_len*len(self.attribute_dims),  self.hidden]
        k=k.transpose(1, 2) # [batch_size, self.hidden, self.max_seq_len*len(self.attribute_dims)]

        attention_scores= torch.bmm(q, k)  # [batch_size, 1, self.max_seq_len*len(self.attribute_dims)]
        attention_scores=attention_scores/ math.sqrt(self.hidden)

        mask=mask.unsqueeze(1)
        num_attr = int(attention_scores.shape[2]/mask.shape[2])
        mask = mask.repeat((1, 1,num_attr))

        attention_scores[mask] = float('-inf')

        attention_probs = nn.Softmax(dim=-1)(attention_scores) #[ batch_size, 1, self.max_seq_len*len(self.attribute_dims)]

        result = torch.bmm(attention_probs, enc_output).transpose(0, 1)  # [1, batch_size, enc_hid_dim]

        return result

class Decoder_act(nn.Module):
    def __init__(self, vocab_size, hid_dim,num_layers,output_dim):
        super().__init__()
        self.num_layers=num_layers
        self.vocab_size = vocab_size
        self.attention =  Attention(hid_dim, hid_dim)
        self.embedding = nn.Embedding(vocab_size, hid_dim)
        self.rnn = nn.GRU( 2*hid_dim, hid_dim,num_layers = num_layers,dropout=0.3)
        self.fc_out = nn.Linear(3*hid_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, dec_input, s, enc_output,mask):
        # dec_input = [batch_size]
        # s = [batch_size, hid_dim]
        # enc_output = [case_len*num_attr, 1, hid_dim ]

        dec_input = dec_input.unsqueeze(0) # dec_input = [batch_size]=> [1,batch_size]
        dec_input =self.embedding(dec_input) # dec_input = [1,batch_size] => [1,batch_size,hid_dim]

        dropout_dec_input = self.dropout(dec_input) #  [1, batch_size,hid_dim]

        # c = [1, batch_size, hid_dim]
        c = self.attention(s, enc_output,mask)

        rnn_input = torch.cat((dropout_dec_input, c), dim = 2) # rnn_input = [1, batch_size, hid_dim+ hid_dim]

        # dec_output=[1,batch_size,dec_hid_dim]  ; dec_hidden=[num_layers,batch_size,hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.repeat( self.num_layers,1,1))

        dec_output = dec_output.squeeze(0) # dec_output:[ batch_size, hid_dim]

        c = c.squeeze(0)  # c:[1, batch_size, hid_dim] => [batch_size, hid_dim]

        dropout_dec_input=dropout_dec_input.squeeze(0)  # dropout_dec_input:[1, batch_size, hid_dim] => [batch_size, hid_dim]

        pred = self.fc_out(torch.cat((dec_output, c, dropout_dec_input), dim = 1))# pred = [batch_size, output_dim]

        return pred, dec_hidden[-1]

class Decoder_attr(nn.Module):
    def __init__(self, vocab_size, hid_dim,num_layers,output_dim,TF_styles):
        super().__init__()
        self.num_layers=num_layers
        self.vocab_size = vocab_size
        self.attention =  Attention(hid_dim, hid_dim)
        self.embedding_act = nn.Embedding(vocab_size, hid_dim)
        self.embedding_attr = nn.Embedding(output_dim, hid_dim)
        self.TF_styles=TF_styles
        emb_num = 1
        if TF_styles == 'FAP' :
            emb_num=2
        self.rnn = nn.GRU(hid_dim * emb_num + hid_dim, hid_dim,num_layers = num_layers,dropout=0.3)
        self.fc_out = nn.Linear(hid_dim  + hid_dim + hid_dim * emb_num , output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, dec_input_act,dec_input_attr, s, enc_output,mask):
        # dec_input_act = [batch_size]
        # dec_input_attr = [batch_size]
        # s = [batch_size, hid_dim]
        # enc_output = [case_len*num_attr, 1, hid_dim * 2]

        dec_input_act = dec_input_act.unsqueeze(0) # dec_input = [1]=> [1,batch_size]
        dec_input_act =self.embedding_act(dec_input_act) # dec_input = [1, batch_size] => [1, batch_size,hid_dim]

        dropout_dec_input_act = self.dropout(dec_input_act) #  [1, batch_size,hid_dim]

        dec_input_attr = dec_input_attr.unsqueeze(0)  # dec_input = [1, batch_size]
        dec_input_attr = self.embedding_attr(dec_input_attr)  # dec_input = [1, batch_size] => [1, batch_size,hid_dim]

        dropout_dec_input_attr = self.dropout(dec_input_attr)  # [1, batch_size,hid_dim]

        # c = [1, batch_size, hid_dim]
        c = self.attention(s, enc_output,mask)

        if  self.TF_styles=='AN':
            rnn_input = torch.cat((dropout_dec_input_act,  c),
                                  dim=2)  # rnn_input = [1, batch_size, hid_dim + hid_dim]
        elif  self.TF_styles=='PAV':
            rnn_input = torch.cat((dropout_dec_input_attr, c),
                                  dim=2)  # rnn_input = [1, batch_size, hid_dim+ hid_dim]
        else:  #FAP
            rnn_input = torch.cat((dropout_dec_input_act, dropout_dec_input_attr, c),
                                  dim=2)  # rnn_input = [1, batch_size, (hid_dim * 2)+ hid_dim]


        dec_output, dec_hidden = self.rnn(rnn_input, s.repeat( self.num_layers,1,1))
        # dec_output=[1,batch_size,hid_dim]  ; dec_hidden=[num_layers,batch_size,hid_dim]
        dec_output = dec_output.squeeze(0) # dec_output:[ batch_size, hid_dim]

        c = c.squeeze(0)  # c:[batch_size, hid_dim]

        dropout_dec_input_act=dropout_dec_input_act.squeeze(0)  # dropout_dec_input_act:[batch_size, hid_dim]
        dropout_dec_input_attr=dropout_dec_input_attr.squeeze(0) # dropout_dec_input_attr:[batch_size, hid_dim]

        if self.TF_styles == 'AN':
            pred = self.fc_out(torch.cat((dec_output, c, dropout_dec_input_act), dim = 1)) # pred = [batch_size, output_dim]
        elif self.TF_styles == 'PAV':
            pred = self.fc_out(torch.cat((dec_output, c,  dropout_dec_input_attr), dim=1)) # pred = [batch_size, output_dim]
        else:  # FAP
            pred = self.fc_out(torch.cat((dec_output, c, dropout_dec_input_act,dropout_dec_input_attr), dim = 1)) # pred = [batch_size, output_dim]

        return pred, dec_hidden[-1]

class GAT_AE(nn.Module):
    def __init__(self,  attribute_dims,max_seq_len ,hidden_dim, GAT_heads, decoder_num_layers,TF_styles):
        super().__init__()
        encoders=[]
        decoders=[]
        self.max_seq_len=max_seq_len
        self.attribute_dims=attribute_dims
        for i, dim in enumerate(attribute_dims):
            encoders.append( GAT_Encoder(int(dim), hidden_dim, GAT_heads, max_seq_len ))
            if i == 0:
                decoders.append(Decoder_act(int(attribute_dims[0] + 1), hidden_dim, decoder_num_layers,
                                            int(dim + 1)))
            else:
                decoders.append(
                    Decoder_attr(int(attribute_dims[0] + 1), hidden_dim, decoder_num_layers,
                                 int(dim + 1),TF_styles))
        self.encoders=nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)


    def forward(self, graphs , Xs, mask,batch_size):
        '''
        :param graphs:是多个属性对应的图，每一个属性作为一个graph
        :param Xs:是多个属性，每一个属性作为一个X ： 列表长度为len(attribute_dims)，列表中元素长度为 [batch_size,seq_len]
        :param mask:(batch_size,seq_len)
        :param batch_size:
        :return:
        '''
        attr_reconstruction_outputs = [] #概率分布 probability map
        s = []  #解码层GRU初始隐藏表示
        enc_output = None
        # Z=None
        for i, dim in enumerate(self.attribute_dims):
            output_dim = int(dim) + 1
            graph = graphs[i]

            if graph.is_cuda:
                attr_reconstruction_outputs.append(torch.zeros(self.max_seq_len, batch_size, output_dim).cuda())  # 存储decoder的所有输出
            else:
                attr_reconstruction_outputs.append(torch.zeros(self.max_seq_len, batch_size, output_dim))
            enc_output_ = self.encoders[i](graph,batch_size) # enc_output_:[batch_size, self.max_seq_len , hidden_dim]
            enc_output_ = enc_output_.permute((1,0,2)) # enc_output_:[self.max_seq_len ,batch_size , hidden_dim]
            s_= enc_output_.mean(0)  #取所有节点的平均作为decoder的第一个隐藏状态的输入  s_:[batch_size, hidden_dim]
            if enc_output is None:
                enc_output = enc_output_
            else:
                enc_output = torch.cat((enc_output, enc_output_), dim=0)
            # enc_output = [self.max_seq_len*len(self.attribute_dims), batch_size, hidden_dim ]
            s.append(s_)


        for i, dim in enumerate(self.attribute_dims):
            if i == 0:
                X = Xs[i]   #[batch_size, self.max_seq_len]
                s0 = s[i]  # s0 :[batch_size, hidden_dim]
                dec_input = X[:,0] # target的第一列，即是起始字符 teacher_forcing   [batch_size]

                for t in range(1,  self.max_seq_len):
                    dec_output, s0 = self.decoders[i](dec_input, s0, enc_output,mask)

                    # 存储每个时刻的输出
                    attr_reconstruction_outputs[i][t] = dec_output #dec_output:[batch_size,output_dim]

                    dec_input = X[:,t] # teacher_forcing
            else:
                s0 = s[i] # s0 :[batch_size, hidden_dim]
                X_act = Xs[0]  # activity  [batch_size, self.max_seq_len]
                X_attr = Xs[i] #  [batch_size, self.max_seq_len]
                dec_input_attr = X_attr[:,0]  # target的第一列，即是起始字符 teacher_forcing  [batch_size]

                for t in range(1,  self.max_seq_len):
                    dec_input_act = X_act[:,t]  # teacher_forcing activity [batch_size]

                    dec_output, s0 = self.decoders[i](dec_input_act, dec_input_attr, s0, enc_output,mask)  # s0隐藏状态

                    # 存储每个时刻的输出
                    attr_reconstruction_outputs[i][t] = dec_output  #dec_output:[batch_size,output_dim]

                    dec_input_attr = X_attr[:,t] # teacher_forcing [batch_size]

        for i,attr_reconstruction_output in enumerate(attr_reconstruction_outputs):
            attr_reconstruction_outputs[i] = attr_reconstruction_output.transpose(0, 1)

        return attr_reconstruction_outputs
