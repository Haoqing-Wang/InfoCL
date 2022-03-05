import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        if self.downsample: 
            self.ds_conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
            self.ds_bn1 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.ds_bn1(self.ds_conv1(x))
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if self.downsample:
            self.ds_conv1 = nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
            self.ds_bn1 = nn.BatchNorm2d(planes * self.expansion)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            residual = self.ds_bn1(self.ds_conv1(x))

        out += residual
        out = self.relu(out)
        return out

class ResNet_imgnet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet_imgnet, self).__init__()
        self.inplanes = 64       
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)            
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.feat_dim = 512 * block.expansion
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = False
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = True
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)    
        x = self.relu(x)
        x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        out2 = self.avgpool(layer2).squeeze()
        layer3 = self.layer3(layer2)
        out3 = self.avgpool(layer3).squeeze()
        layer4 = self.layer4(layer3)
        out4 = self.avgpool(layer4).squeeze()
        return out2, out3, out4


def resnet18_imagenet():
    return ResNet_imgnet(BasicBlock, [2, 2, 2, 2])

def resnet34_imagenet():
    return ResNet_imgnet(BasicBlock, [3, 4, 6, 3])

def resnet50_imagenet():
    return ResNet_imgnet(Bottleneck, [3, 4, 6, 3])

def resnet101_imagenet():
    return ResNet_imgnet(Bottleneck, [3, 4, 23, 3])

def resnet152_imagenet():
    return ResNet_imgnet(Bottleneck, [3, 8, 36, 3])