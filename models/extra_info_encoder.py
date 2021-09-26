import torch
import torch.nn as nn



class ExtraInfoEncoder(nn.Module):
    
    def __init__(self, n_in, n_out, hidden_layers):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.hidden_layers = hidden_layers
        self.encoder = nn.ModuleList()
        
        _inp = n_in
        for i, dim in enumerate(hidden_layers):
            if i == 0:
                self.encoder.append(nn.Linear(n_in, dim))
            else:
                self.encoder.append(nn.Linear(hidden_layers[i-1], dim))
            _inp = dim
        self.encoder.append(nn.Linear(_inp, n_out))
        
    def forward(self, x):
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < len(self.encoder) - 1:
                x = torch.relu(x)
        return x
    
if __name__ == "__main__":
    input = torch.rand((2, 10))
    model = ExtraInfoEncoder(10, 128, [])
    outp = model(input)
    print(outp.shape)