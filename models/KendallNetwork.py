import torch
import torch.nn as nn


class KendallNetwork(nn.Module):

    def __init__(self, flatten=True):
        super().__init__()
        self.flatten = flatten
        self.kernels_size = [4, 4, 4, 4]
        self.strides = [2, 2, 2, 2]
        self.paddings = [0, 0, 0, 0]
        self.dilations = [1, 1, 1, 1]
        self.out_channel = 256

        self.conv = nn.Sequential(nn.Conv2d(3, 32, kernel_size=4, stride=2),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 128, kernel_size=4, stride=2),
                                 nn.ReLU(),
                                 nn.Conv2d(128, 256, kernel_size=4, stride=2),
                                 nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        if self.flatten:
            x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":

    model = KendallNetwork()
    tensor = torch.randn(2, 3, 256, 256)
    outp = model(tensor)
    print(outp.shape)
