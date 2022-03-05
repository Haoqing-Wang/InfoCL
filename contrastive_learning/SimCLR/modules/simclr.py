import torch.nn as nn
import torchvision
from .resnet import *
from .resnet_imagenet import *


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet to obtain hi = f(xi) = ResNet(xi) where hi is the output after the average pooling layer.
    """
    def __init__(self, args, data='non_imagenet'):
        super(SimCLR, self).__init__()
        self.args = args
        if data == 'imagenet':
            self.encoder = self.get_imagenet_resnet(args.resnet)
        else:
            self.encoder = self.get_resnet(args.resnet)

        self.n_features = self.encoder.feat_dim
        self.projector = nn.Sequential(nn.Linear(self.n_features, self.n_features),
                                       nn.ReLU(),
                                       nn.Linear(self.n_features, args.projection_dim))

    def get_resnet(self, name):
        resnets = {
            "resnet18": resnet18(data=self.args.dataset),
            "resnet34": resnet34(data=self.args.dataset),
            "resnet50": resnet50(data=self.args.dataset),
            "resnet101": resnet101(data=self.args.dataset),
            "resnet152": resnet152(data=self.args.dataset)}
        if name not in resnets.keys():
            raise KeyError(f"{name} is not a valid ResNet version")
        return resnets[name]
     
    def get_imagenet_resnet(self, name):
        resnets = {
            "resnet18": resnet18_imagenet(),
            "resnet34": resnet34_imagenet(),
            "resnet50": resnet50_imagenet(),
            "resnet101": resnet101_imagenet(),
            "resnet152": resnet152_imagenet()}
        if name not in resnets.keys():
            raise KeyError(f"{name} is not a valid ResNet version")
        return resnets[name]

    def forward(self, x):
        if self.args.model == 'LBE':
            mu2, mu3, mu4 = self.encoder(x)
            esp2 = mu2.data.new(mu2.size()).normal_(0., self.args.zeta)
            h2 = mu2 + esp2
            esp3 = mu3.data.new(mu3.size()).normal_(0., self.args.zeta)
            h3 = mu3 + esp3
            esp4 = mu4.data.new(mu4.size()).normal_(0., self.args.zeta)
            h4 = mu4 + esp4
            z = self.projector(h4)
            if self.args.normalize:
                z = nn.functional.normalize(z, dim=1)
            out = (mu2, mu3, mu4, h2, h3, h4, z)
        elif self.args.model == 'MIB':
            _, _, mu = self.encoder(x)
            esp = mu.data.new(mu.size()).normal_(0., self.args.zeta)
            h = mu + esp
            z = self.projector(h)
            if self.args.normalize:
                z = nn.functional.normalize(z, dim=1)
            out = (mu, h, z)
        else:
            _, _, h = self.encoder(x)
            z = self.projector(h)
            if self.args.normalize:
                z = nn.functional.normalize(z, dim=1)
            out = (h, z)
        return out