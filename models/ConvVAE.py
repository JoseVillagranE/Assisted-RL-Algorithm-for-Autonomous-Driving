#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 00:50:11 2021

@author: josev
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils.Network_utils import freeze_params

from LSTM import LSTM, MDN_RNN


class VAE_Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        num_actions,
        n_channel,
        z_dim,
        beta=1.0,
        VAE_weights_path="",
        temporal_mech=False,
        rnn_config=None,
        linear_layers=None,
        is_freeze_params=True,
        wp_encode=False,
        wp_encoder_size=64,
        n_hl_actions=0,
    ):

        super().__init__()
        self.num_actions = num_actions
        self.vae = ConvVAE(n_channel, z_dim, beta=beta)
        self.n_hl_actions = n_hl_actions
        self.softmax = nn.Softmax()

        self.lstm = None
        if temporal_mech:
            if rnn_config["rnn_type"] == "lstm":
                self.lstm = LSTM(
                    input_size=rnn_config["input_size"],
                    hidden_size=rnn_config["hidden_size"],
                    num_layers=rnn_config["num_layers"],
                )
            elif rnn_config["rnn_type"] == "mdn_rnn":
                self.lstm = MDN_RNN(
                    input_size=rnn_config["input_size"],
                    hidden_size=rnn_config["hidden_size"],
                    seq_len=rnn_config["n_steps"],
                    batch_size=rnn_config["batch_size"],
                    device=rnn_config["device"],
                    action_size=num_actions,
                    num_layers=rnn_config["num_layers"],
                    gaussians=rnn_config["gaussians"],
                    mode="inference",
                )
            else:
                raise NotImplementedError(
                    "Only lstm and mdn_rnn type of recurrent models at moment"
                )

        if self.lstm:
            input_linear_layer_dim = state_dim * rnn_config["n_steps"]
        else:
            input_linear_layer_dim = state_dim

        self.mlp = nn.ModuleList()

        if len(linear_layers) > 0:
            for i, dim_layer in enumerate(linear_layers):
                if i == 0:
                    self.mlp.append(nn.Linear(input_linear_layer_dim, dim_layer))
                else:
                    self.mlp.append(nn.Linear(linear_layers[i - 1], dim_layer))
                    
            if self.n_hl_actions == 0:
                self.mlp.append(nn.Linear(linear_layers[-1], num_actions))

        else:
            if self.n_hl_actions == 0:
                self.mlp.append(nn.Linear(input_linear_layer_dim, num_actions))

        if wp_encode:
            self.wp_encoder = nn.Linear(1, wp_encoder_size)

        if len(VAE_weights_path) > 0:
            print("Loading VAE weights..")
            self.vae.load_state_dict(torch.load(VAE_weights_path))
            if is_freeze_params:
                print("Freezing VAE")
                freeze_params(self.vae)

        if self.lstm is not None:
            print("Loading RNN weights..")
            self.lstm.load_state_dict(
                torch.load(rnn_config["weights_path"]), strict=False
            )
            if is_freeze_params:
                print("Freezing RNN")
                freeze_params(self.lstm)

        if self.n_hl_actions == 0:
            self.linear_forward = self._linear_forward
        else:

            self.n_actions_params = num_actions - self.n_hl_actions
            
            self.action_parameters_passthrough_layer = nn.Linear(
                input_linear_layer_dim, self.n_actions_params
            )

            if len(linear_layers) > 0:
                input_linear_layer_dim = linear_layers[-1]

            self.linear_forward = self._hrl_linear_forward
            self.action_outp_layer = nn.Linear(
                input_linear_layer_dim, self.n_hl_actions
            )
            self.action_parameters_outp_layer = nn.Linear(
                input_linear_layer_dim, self.n_actions_params
            )
            

    def forward(self, state):

        """
        state: (B, 1, Z_dim) # Seq_len = 1
        if lstm: state : (B, n_steps, Z_dim+Act) # Seq_len = n_steps
        """
        input_linear = state
        if self.lstm:
            next_state = self.lstm(state)  # (B, Z_dim+Compl_State)
            input_linear = torch.cat(
                (state[:, -1, :].squeeze(1), next_state), dim=-1
            ).squeeze(0)
        action = self.linear_forward(input_linear)
        return action

    def _linear_forward(self, x):
        for i, layer in enumerate(self.mlp):
            if i == len(self.mlp) - 1:
                x = torch.tanh(layer(x))
            else:
                x = torch.relu(layer(x))
        return x

    def _hrl_linear_forward(self, state):
        x = state
        for i, layer in enumerate(self.mlp):
            x = torch.relu(layer(x))
        probs = self.action_outp_layer(x)
        p_actions = self.action_parameters_outp_layer(x)
        # p_actions += self.action_parameters_passthrough_layer(state)
        return (probs, p_actions)

    def feat_ext(self, x):
        """
        Parameters
        ----------
        state : PIL Image
            image from RGB camera.
        Returns
        -------
        z: latent state.
        """
        mu, logvar = self.vae.encode(x)
        return self.vae.reparametrize(mu, logvar)

    def wp_encode_fn(self, wp):
        wp = torch.tensor(wp).unsqueeze(0).float()
        return self.wp_encoder(wp).detach().numpy()


class VAE_Critic(nn.Module):  # No needed temporal mechanism
    def __init__(self, 
                 state_dim,
                 num_actions,
                 hidden_layers=[],
                 temporal_mech=False,
                 rnn_config={},
                 is_freeze_params=True
                 ):
        super().__init__()
        self.num_actions = num_actions
        self.layers = nn.ModuleList()
        
        self.lstm = None
        if temporal_mech:
            if rnn_config["rnn_type"] == "lstm":
                self.lstm = LSTM(
                    input_size=rnn_config["input_size"],
                    hidden_size=rnn_config["hidden_size"],
                    num_layers=rnn_config["num_layers"],
                )
            elif rnn_config["rnn_type"] == "mdn_rnn":
                self.lstm = MDN_RNN(
                    input_size=rnn_config["input_size"],
                    hidden_size=rnn_config["hidden_size"],
                    seq_len=rnn_config["n_steps"],
                    batch_size=rnn_config["batch_size"],
                    device=rnn_config["device"],
                    action_size=num_actions,
                    num_layers=rnn_config["num_layers"],
                    gaussians=rnn_config["gaussians"],
                    mode="inference",
                )
            else:
                raise NotImplementedError(
                    "Only lstm and mdn_rnn type of recurrent models at moment"
                )
                
        if self.lstm is not None:
            print("Loading RNN weights..")
            self.lstm.load_state_dict(
                torch.load(rnn_config["weights_path"]), strict=False
            )
            if is_freeze_params:
                print("Freezing RNN")
                freeze_params(self.lstm)
        
        if self.lstm:
            input_dim = state_dim * rnn_config["n_steps"] + num_actions
        else:
            input_dim = state_dim + num_actions
        
        
        for i, outp_dim_layer in enumerate(hidden_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, outp_dim_layer))
            else:
                self.layers.append(nn.Linear(hidden_layers[i-1], outp_dim_layer))
        
        if len(hidden_layers) > 0:
            self.layers.append(nn.Linear(hidden_layers[-1], 1))
        else:
            self.layers.append(nn.Linear(input_dim, 1))

    def forward(self, state, action):
        
        _state = state
        if self.lstm:
            next_state = self.lstm(state)  # (B, Z_dim+Compl_State)
            _state = torch.cat(
                (state[:, -1, :].squeeze(1), next_state), dim=-1
            ).squeeze(0)
        x = torch.cat([_state, action], 1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class ConvVAE(nn.Module):
    def __init__(self, n_channel, z_dim, beta=1.0, reduction="sum", recons_loss="bce"):
        super().__init__()

        self.n_channel = n_channel
        self.z_dim = z_dim
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Conv2d(n_channel, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        self.decoder_input = nn.Linear(z_dim, 256 * 24)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_channel, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )

        self.mu = nn.Linear(256 * 24, z_dim)
        self.logvar = nn.Linear(256 * 24, z_dim)

        if recons_loss == "bce":
            self.recons_loss = nn.BCELoss(reduction=reduction)
        else:
            self.recons_loss = nn.MSELoss(reduction=reduction)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recons = self.decode(z)
        return recons, mu, logvar

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)  # (N, C, H, W)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar, mode="testing"):

        sigma = torch.exp(0.5 * logvar)
        eps = torch.rand_like(sigma)
        z = mu + sigma * eps
        return z

    def decode(self, z):
        z = self.decoder_input(z).view(-1, 256, 3, 8)
        recons = self.decoder(z)
        return recons

    def compute_loss(self, input, recons, mu, logvar):
        recons_loss = self.recons_loss(recons, input)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        return recons_loss + self.beta * kl_loss

    def save_struct(self, module, file):
        torch.save(getattr(self, module).state_dict(), file)

    def load_struct(self, module, path):
        getattr(self, module).load_state_dict(torch.load(path))

    def to_device(self, module, device):
        getattr(self, module).to(device)


if __name__ == "__main__":

    model = ConvVAE(3, 64, beta=1.0, reduction="sum", recons_loss="mse")
    input = torch.rand((10, 3, 80, 160))

    recons, mu, logvar = model(input)
    loss = model.compute_loss(input, recons, mu, logvar)
    encoder = model.to_device("encoder", "cpu")
