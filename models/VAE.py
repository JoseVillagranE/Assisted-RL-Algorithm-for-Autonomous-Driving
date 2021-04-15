#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 00:50:11 2021

@author: josev
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class VAE_Actor(nn.Module):
        
    def __init__(self, 
                 state_dim,
                 num_actions,
                 n_channel,
                 z_dim,
                 beta=1.0,
                 weights_path="",
                 freeze_params=False):
        super().__init__()        
        self.num_actions = num_actions
        self.vae = ConvVAE(n_channel, z_dim, beta=beta)
        self.linear = nn.Linear(state_dim, num_actions)
        if len(weights_path) > 0:
            self.vae.load_state_dict(torch.load(weights_path))
            if freeze_params:
                freeze_params(self.vae)
        
        
    def forward(self, state):
        x = self.linear(state)
        action = torch.tanh(x) 
        return action
    
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
    
class VAE_critic(nn.Module):
    
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.linear = nn.Linear(state_dim+num_actions, 1)
        
    def forward(self, state, action):
        return self.linear(torch.cat([state, action], 1))

class ConvVAE(nn.Module):
    
    def __init__(self, n_channel, z_dim, beta=1.0, reduction="sum", recons_loss="bce"):
        super().__init__()
        
        self.n_channel = n_channel
        self.z_dim = z_dim
        self.beta = beta
        
        self.encoder = nn.Sequential(nn.Conv2d(n_channel, 32, kernel_size=4, stride=2), nn.ReLU(),
                                     nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                                     nn.Conv2d(64, 128, kernel_size=4, stride=2), nn.ReLU(),
                                     nn.Conv2d(128, 256, kernel_size=4, stride=2), nn.ReLU())
        
        self.decoder_input = nn.Linear(z_dim, 256*24)
        
        self.decoder = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2), nn.ReLU(),
                                     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2), nn.ReLU(),
                                     nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2), nn.ReLU(),
                                     nn.ConvTranspose2d(32, n_channel, kernel_size=4, stride=2), nn.Sigmoid())
        
        self.mu = nn.Linear(256*24, z_dim)
        self.logvar = nn.Linear(256*24, z_dim)
        
        self.recons_loss = nn.BCELoss(reduction=reduction) if recons_loss=="bce" \
                                                            else nn.MSELoss(reduction=reduction)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recons = self.decode(z)
        return recons, mu, logvar
    
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1) # (N, C, H, W)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    
    def reparametrize(self, mu, logvar):
        
        sigma = torch.exp(0.5*logvar)
        eps = torch.rand_like(sigma)
        z = mu + sigma*eps
        return z
        
    
    def decode(self, z):
        z = self.decoder_input(z).view(-1, 256, 3, 8)
        recons = self.decoder(z)
        return recons
    
    def compute_loss(self, input, recons, mu, logvar):
        recons_loss = self.recons_loss(recons, input)
        kl_loss = -0.5*torch.sum(1 + logvar - mu**2 - logvar.exp())
        return recons_loss + self.beta*kl_loss
    
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
    