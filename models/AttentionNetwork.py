import torch
import torch.nn as nn
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from AlexNet import alexnet, AlexNet
from DDPG import conv2d_size_out

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, H=24, W=24, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstmcell = nn.LSTMCell(input_size, hidden_size, num_layers)
        self.c_0 = nn.Linear(input_size, hidden_size) # (D -> hidden)
        self.h_0 = nn.Linear(input_size, hidden_size)

    def forward(self, input, hidden_state):
        """
        :param input: (Tensor: [B, input_size])
        :param hidden_state: (Tensor: [B, Hidden_size])
        :return h_n: (Tensor: [B, Hidden_size])
        """
        (h_n, c_n) = self.lstmcell(input, hidden_state)
        return (h_n, c_n)

    def init_hidden_state(self, x_0):
        """
        :parama x_0: (Tensor [B, num_pix, D]) batch w/ initial feature slices
        :return h_0: (Tensor: [B, Hidden_size]) initial hidden states for the first features slices
        :return c_0: (Tensor: [B, Hidden_size])
        """
        h_0 = self.h_0(x_0.mean(dim=1).squeeze()) # size of the input [B, D]
        c_0 = self.c_0(x_0.mean(dim=1).squeeze())
        return (h_0, c_0)

class Decoder_Attention(nn.Module):

    def __init__(self, input_size, hidden_size, T, H=24, W=24):

        """
        :param input_size: (int) Feature size of each pixel in the feat. cube
        :param hidden_size: (int) Dimension of hidden state
        """

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn_layer = nn.Linear(input_size+hidden_size, 1)
        self.lstm = LSTM(input_size, hidden_size, H=H, W=W)# [T, B, num_dir*hidden_size] [num_lay*num_dir, B, hidden_size]
        self.linear_outp = nn.Conv1d(T, 1, kernel_size=1, bias=False)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def forward(self, dec_in):
        """
        :param dec_in: (Tensor: [T, B, D, H_out, W_out]) Batch of features cubes separates in windows of time
        """
        B, T, D, H_out, W_out = dec_in.size()
        dec_in = dec_in.view(T, B, D, -1).transpose(2,3) #[T, B, num_pix, D]
        inp_hidden = dec_in[0, :, :, :].squeeze() # [B, num_pix, D]
        # h = self.lstm.init_hidden_state(inp_hidden) # [B, Hidden_size]
        h = torch.zeros((B, self.hidden_size)).to(device)
        c = torch.zeros((B, self.hidden_size)).to(device)
        predictions = torch.zeros((T, B))
        alphas = torch.zeros((T, B, H_out, W_out))

        h_list = []
        g_t_list = []

        for t in range(T):
            g_t = self.attn_layer(torch.cat((dec_in[t, :, :, :].squeeze(),
                                            h.unsqueeze(1).expand(-1, H_out*W_out, -1)),
                                            dim=2)
                                            ) # [B, num_pix, 1]

            g_t = F.softmax(g_t) # [B, num_pix, 1]
            z_t = (dec_in[t, :, :, :].squeeze() * g_t).sum(dim=1) # [B, D]
            (h, c) = self.lstm(z_t, (h, c)) # [B, hidden]
            h_list.append(h)
            g_t_list.append(g_t.squeeze(-1))

        h = torch.stack(h_list, dim=1)# [batch, T, hidden]
        outp = self.linear_outp(h) # [B, hidden]
        g_t = torch.stack(g_t_list, dim=-1) # [B, num_pix, T]
        return outp, g_t

class AttentionNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, T, num_layers=1, H_in=80, W_in=160):

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.H_in = H_in
        self.W_in = W_in
        pretrained = False

        self.alexnet_model = alexnet(pretrained, flatten=False) # feature extractor

        H_out =  conv2d_size_out(H_in,
                                self.alexnet_model.kernels_size,
                                self.alexnet_model.strides,
                                self.alexnet_model.paddings,
                                self.alexnet_model.dilations,
                                )
        W_out =  conv2d_size_out(W_in,
                                self.alexnet_model.kernels_size,
                                self.alexnet_model.strides,
                                self.alexnet_model.paddings,
                                self.alexnet_model.dilations,
                                )

        self.dec_attention = Decoder_Attention(input_size, hidden_size, T, H=H_out, W=W_out)

    def forward(self, x):
        """
        :param x: (Tensor: [B, T, C, H, W]) Batch of Images separates for windows of time
        :return loss:(Tensor) Loss function for train the network
        """

        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size*timesteps, C, H, W)
        x_t = self.alexnet_model(x) # feature cube -> x_t : (B*Timestamp, D, H_out, W_out)
        x_t = x_t.view(batch_size, timesteps, x_t.shape[1], x_t.shape[2], x_t.shape[3]) # [B, T, D, H_out, W_out]
        predictions, attention_weights = self.dec_attention(x_t) #
        return predictions.squeeze(), attention_weights



if __name__ == "__main__":

    batch_size = 64
    timestep = 10
    C = 3
    H = 240
    W = 240
    input_size = 256 # C_out from AlexNet
    hidden_size = 128

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    batch = torch.rand((batch_size, timestep, C, H, W)).to(device)
    model = AttentionNetwork(input_size, hidden_size, timestep, H_in=H, W_in=W).to(device)

    predictions, attention_weights = model(batch) # [batch, hidden], [batch, H*W, T]

    print(predictions.shape)
    print(attention_weights.shape)
