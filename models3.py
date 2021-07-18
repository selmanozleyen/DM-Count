import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

__all__ = ['vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
        )
        # self.into_mu = nn.ReLU(inplace=True)

        # self.into_v = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Softmax(dim=1),
        # )
        # self.reg_layer2 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(1024*2,1024*3//2),
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        #     nn.Linear(1024*3//2,1024*3//2),
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        #     nn.Linear(1024*3//2,1024),
        #     nn.Dropout(0.2),
        #     nn.ReLU(),
        #     nn.Linear(1024,1024),
        #     nn.Dropout(0.2),
        #     nn.Linear(1024,1024),
        #     nn.Dropout(0.2),
        #     nn.Linear(1024,1024),
        #     nn.Dropout(0.2),
        #     nn.Linear(1024,1024),
        #     nn.Dropout(0.2),
        #     nn.ReLU(),
        #     nn.Linear(1024,1024),
        #     nn.Softmax(dim=1)
        # )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())
        self.density_layer2 = nn.Sequential(nn.Conv2d(128, 1, 1),nn.Flatten(),nn.ReLU())

    def forward(self, x):
        x = self.features(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)
        v = self.density_layer2(x)
        mu = self.density_layer(x)
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed, v

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19_3():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model

# class D(nn.Module):
#     def __init__(self):
#         super(D, self).__init__()
#         self.
#     def forward(self,x,y):
