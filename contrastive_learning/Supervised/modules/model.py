import torch.nn as nn
import torchvision
from .resnet import *

class Pretrain(nn.Module):
    def __init__(self, args, nclass):
        super(Pretrain, self).__init__()
        self.args = args
        self.encoder = self.get_resnet(args.resnet)
        self.n_features = self.encoder.feat_dim
        self.nclass = nclass
        self.predictor = nn.Linear(self.n_features, self.nclass, bias=True)

    def get_resnet(self, name):
        resnets = {
            "resnet18": resnet18(),
            "resnet34": resnet34(),
            "resnet50": resnet50(),
            "resnet101": resnet101(),
            "resnet152": resnet152()}
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
            y = self.predictor(h4)
            out = (mu2, mu3, mu4, h2, h3, h4, y)
        else:
            _, _, h = self.encoder(x)
            y = self.predictor(h)
            out = (h, y)
        return out