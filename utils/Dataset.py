import glob
import os
import numpy as np
from bisect import bisect
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile


def to_latent(vae_model, states, next_states, complt_states=None, next_complt_states=None):
    zs = []
    next_zs = []
    for i in range(states.shape[1]):
        _, mu, logvar = vae_model(states[:, i, :, :, :].squeeze())
        _, next_mu, next_logvar = vae_model(next_states[:, i, :, :, :].squeeze())
        z = vae_model.reparametrize(mu, logvar).unsqueeze(1)
        next_z = vae_model.reparametrize(next_mu, next_logvar).unsqueeze(1)
        zs.append(z)
        next_zs.append(next_z)
    states = torch.cat(zs, axis=1) # (B, S, Z_dim)
    next_states = torch.cat(next_zs, axis=1) # (B, S, Z_dim)
    if complt_states:
        complt_states = torch.from_numpy(complt_states)
        next_complt_states = torch.from_numpy(next_complt_states)
        states = torch.cat((states, complt_states), axis=2) # (B, S, Z_dim+compl_dim)
        next_states = torch.cat((next_states, next_complt_states), axis=2) # (B, S, Z_dim+compl_dim)
    return states, next_states

class Rollout_Dataset(Dataset):
    
    
    def __init__(self, path, idxs, seq_len, buffer_size, is_complt_states, transform=None):
        
        self._files = glob.glob(os.path.join(path, "*.npz"))
        self._files = sorted(self._files, key=lambda name: int(name.split('_')[-3]))
        self._files = [self._files[idx] for idx in idxs]
        self._buffer_size = buffer_size
        self._seq_len = seq_len
        self._transform = transform
        self.is_complt_states = is_complt_states
        
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._cum_size = [0]
    
    def __len__(self):
        if self._cum_size:
            self.load_next_data_buffer()
        return self._cum_size[-1]
    
    def __getitem__(self, idx):
        file_index = bisect(self._cum_size, idx) - 1
        seq_index = idx - self._cum_size[file_index]
        data = self._buffer[file_index]
        return self._get_data(data, seq_index)
    
    def load_next_data_buffer(self):
        
        self._buffer_fnames = self._files[self._buffer_index:self._buffer_index+
                                                             self._buffer_size]
        
        self._buffer_index += self._buffer_size
        self._buffer = []
        self._cum_size = [0]
        
        for f in self._buffer_fnames:
            with np.load(f) as data:
                self._buffer += [{k: np.copy(v) for k, v in data.items()}]
                self._cum_size += [self._cum_size[-1] + 
                                   self._data_per_seq(data["rewards"].shape[0])]
    
    def _get_data(self, data, seq_index):
        # obs_data -> [seq_len, C, H, W]
        obs_data = data['states'][seq_index:seq_index+self._seq_len+1].astype(np.float32)
        obs_data = np.transpose(obs_data, (0, 2, 3, 1)) # [S, H, W, C]
        obs_data = [Image.fromarray(np.uint8(obs*255)) for obs in obs_data]
        if self._transform: 
            obs_data = [self._transform(obs).unsqueeze(0) for obs in obs_data]
            obs_data = torch.cat(obs_data, axis=0)
        obs, next_obs = obs_data[:-1], obs_data[1:]
        action = data['actions'][seq_index+1:seq_index+self._seq_len+1].astype(np.float32)
        reward = data["rewards"][seq_index+1:seq_index+self._seq_len+1].astype(np.float32)
        terminal = data["terminals"][seq_index+1:seq_index+self._seq_len+1]
        if self.is_complt_states: 
            complt_states = data['complementary_states'][seq_index:seq_index+self._seq_len+1]
            complt_states = complt_states.astype(np.float32)
            complt_states, next_complt_states = complt_states[:-1], complt_states[1:]
        # obs -> [B, S, C, H, W]
        return obs, action, reward, next_obs, terminal, complt_states, next_complt_states
    
    def _data_per_seq(self, data_length):
        return data_length - self._seq_len
    
    
    
    
