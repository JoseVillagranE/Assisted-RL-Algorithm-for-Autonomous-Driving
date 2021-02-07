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



class ConvVAE(nn.Module):
    
    def __init__(self, n_channel, z_dim, beta=1.0):
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
        
        self.bce = nn.BCELoss(reduction='sum')
        
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
        z = self.decoder_input(z).view(-1, 256, 8, 3)
        recons = self.decoder(z)
        return recons
    
    def compute_loss(self, input, recons, mu, logvar):
        recons_loss = self.bce(recons, input)
        kl_loss = -0.5*torch.sum(1 + logvar - mu**2 - logvar.exp())
        return recons_loss + self.beta*kl_loss
    
    
if __name__ == "__main__":
    
    model = ConvVAE(3, 64, 1.0)
    input = torch.rand((10, 3, 160, 80))
    
    recons, mu, logvar = model(input)
    loss = model.compute_loss(input, recons, mu, logvar)