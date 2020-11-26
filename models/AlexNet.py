import torch
import torch.nn as nn


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, pretrained, num_classes=1000, flatten=True):
        super().__init__()
        self.kernels_size = [11, 3, 5, 3, 3, 3, 3, 3]
        self.strides = [4, 2, 1, 2, 1, 1, 1, 2]
        self.paddings = [2, 0, 2, 0, 1, 1, 1, 0]
        self.dilations = [1, 1, 1, 1, 1, 1, 1, 1]
        self.out_channel = 256
        self.flatten = flatten

        # H and W must be at least 224 of size

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        if pretrained:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )


    def forward(self, x):
        x = self.features(x)
        if self.flatten:
            x = torch.flatten(x, 1)
        return x

def alexnet(pretrained=False, flatten=True, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(pretrained, flatten=flatten)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":

    model = alexnet(False)
    kernels_size = model.kernel_size
    # for name, param in model.named_parameters():
    #     print(name, param.data)
