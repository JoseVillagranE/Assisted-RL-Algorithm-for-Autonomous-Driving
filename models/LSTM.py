import torch
import torch.nn as nn



class LSTM(nn.Module):
    
    def __init__(self, z_dim, h_dim, a_dim):
        super().__init__()
        self.lstm = nn.LSTM(z_dim+a_dim, h_dim, batch_first=True)
        self.linear = nn.Linear(h_dim, z_dim)
        
    def forward(self, state, action):
        """
        x: (batch, seq_len, input_size)
        outp: (batch, seq_len, hidden_size)
        h_n: (batch, num_layers, hidden_size)
        c_n: (batch, num_layers, hidden_size)
        """
        outp, (h_n, c_n) = self.lstm(x)
        return self.linear(h_n)