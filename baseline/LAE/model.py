import math
import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.autograd import Variable
from torchvision import transforms

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.isCuda = isCuda

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # initialize weights
        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, input):
        tt = torch.cuda if self.isCuda else torch
        h0 = Variable(tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size).zero_(), requires_grad=False)
        c0 = Variable(tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size).zero_(), requires_grad=False)
        encoded_input, hidden = self.lstm(input, (h0, c0))
        return encoded_input, hidden


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, isCuda):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.isCuda = isCuda

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

        # initialize weights
        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.linear.weight, gain=np.sqrt(2))

    def forward(self, encoded_input, hidden):
        tt = torch.cuda if self.isCuda else torch
        decoded_output, _ = self.lstm(encoded_input, hidden)
        decoded_output = self.linear(decoded_output)
        return decoded_output


class LSTMAE(nn.Module):
    # x --> lstm --> z --> lstm --> fc --> x'
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(LSTMAE, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, num_layers, isCuda)
        self.decoder = DecoderLSTM(hidden_size, input_size, num_layers, isCuda)

    def forward(self, input):
        encoded_input, hidden = self.encoder(input)
        decoded_output = self.decoder(encoded_input, hidden)
        return decoded_output
