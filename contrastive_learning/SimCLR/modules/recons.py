import torch.nn as nn

class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class ReCon32(nn.Module):
    def __init__(self, indim=512):
        super(ReCon32, self).__init__()
        self.indim = indim
        self.recon = nn.Sequential(
            nn.Linear(indim, 16*64, bias=False),
            nn.BatchNorm1d(num_features=16*64),
            nn.ReLU(inplace=True),
            Lambda(lambda x: x.reshape(-1, 64, 4, 4)),

            nn.ConvTranspose2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True))

    def forward(self, x):
        out = self.recon(x)
        return out

class ReCon64(nn.Module):
    def __init__(self, indim=512):
        super(ReCon64, self).__init__()
        self.indim = indim
        self.recon = nn.Sequential(
            nn.Linear(indim, 16*128, bias=False),
            nn.BatchNorm1d(num_features=16*128),
            nn.ReLU(inplace=True),
            Lambda(lambda x: x.reshape(-1, 128, 4, 4)),

            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True))

    def forward(self, x):
        out = self.recon(x)
        return out

class ReCon224(nn.Module):
    def __init__(self, indim=2048):
        super(ReCon224, self).__init__()
        self.indim = indim
        self.recon = nn.Sequential(
            nn.Linear(indim, 49*64, bias=False),
            nn.BatchNorm1d(num_features=49*64),
            nn.ReLU(inplace=True),
            Lambda(lambda x: x.reshape(-1, 64, 7, 7)),

            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True))

    def forward(self, x):
        out = self.recon(x)
        return out