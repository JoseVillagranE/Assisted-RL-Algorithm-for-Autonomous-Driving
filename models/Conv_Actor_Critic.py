import torch
import torch.nn as nn
import numpy as np

from AlexNet import alexnet, AlexNet
from KendallNetwork import KendallNetwork

class Conv_Actor(nn.Module):

    def __init__(self, num_actions, h_image_in, w_image_in, linear_layers=[],
                    pretrained=False):

        super().__init__()
        self.num_actions= num_actions

        # self.conv_model = alexnet(pretrained) # feature extractor
        self.conv_model = KendallNetwork()

        convh =  conv2d_size_out(h_image_in,
                                self.conv_model.kernels_size,
                                self.conv_model.strides,
                                self.conv_model.paddings,
                                self.conv_model.dilations,
                                )
        convw =  conv2d_size_out(w_image_in,
                                self.conv_model.kernels_size,
                                self.conv_model.strides,
                                self.conv_model.paddings,
                                self.conv_model.dilations,
                                )

        linear_outp_size = convh*convw*self.conv_model.out_channel

        self.linear_layer_list = nn.ModuleList()
        if len(linear_layers) > 0:
            self.linear_layer_list.append(nn.Linear(linear_outp_size, linear_layers[0]))
            for i in range(len(linear_layers) - 1):
                self.linear_layer_list.append(nn.Linear(linear_layers[i], linear_layers[i+1]))

            self.linear_layer_list.append(nn.Linear(linear_layers[-1], num_actions))

        else:
            self.linear_layer_list.append(nn.Linear(linear_outp_size, num_actions))

    def forward(self, state):
        x = self.conv_model(state)
        x = self.forward_linear(x, self.linear_layer_list)
        return x

    @staticmethod
    def forward_linear(x, layer_list):

        for i, layer in enumerate(layer_list):
            if i < len(layer_list) - 1:
                x = torch.relu(layer(x))
            else:
                x = torch.tanh(layer(x)) # just the last layer apply tanh
        return x

class Conv_Critic(nn.Module):

    def __init__(self, action_space,  h_image_in, w_image_in, pretrained=False):
        super().__init__()
        self.action_space = action_space
        self.alexnet_model = alexnet(pretrained) # feature extractor

        convh =  conv2d_size_out(h_image_in,
                                self.alexnet_model.kernels_size,
                                self.alexnet_model.strides,
                                self.alexnet_model.paddings,
                                self.alexnet_model.dilations,
                                )
        convw =  conv2d_size_out(w_image_in,
                                self.alexnet_model.kernels_size,
                                self.alexnet_model.strides,
                                self.alexnet_model.paddings,
                                self.alexnet_model.dilations,
                                )

        linear_outp_size = convh*convw*self.alexnet_model.out_channel
        self.linear = nn.Linear(linear_outp_size + action_space, 1)

    def forward(self, state, action):
        x = self.alexnet_model(state)
        x = torch.cat([x, action], 1)
        x = self.linear(x)
        return x
    
if __name__ == "__main__":
    
    num_actions = 2
    h_image_in = 80
    w_image_in = 160
    linear_layers=[]
    pretrained=False
    
    actor = Conv_Actor(num_actions, h_image_in, w_image_in, linear_layers, pretrained)
    critic = Conv_Critic(num_actions, h_image_in, w_image_in, pretrained)