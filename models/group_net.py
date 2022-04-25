import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import torch.nn.init as init



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, num_bases, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__() 
        self.num_bases = num_bases
        self.relu = nn.ReLU()
        self.conv1 = nn.ModuleList([conv3x3(inplanes, planes, stride) for i in range(num_bases)])       
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(num_bases)])
        self.conv2 = nn.ModuleList([conv3x3(planes, planes) for i in range(num_bases)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(num_bases)])
        self.downsample = downsample
        self.scales = nn.ParameterList([nn.Parameter(torch.rand(1), requires_grad=True) for i in range(num_bases)])

    def forward(self, x):

        final_output = None
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
            

        for conv1, conv2, bn1, bn2, scale in zip(self.conv1, self.conv2, self.bn1, self.bn2, self.scales):

            out = conv1(x)
            out = self.relu(out)
            out = bn1(out)
            out += residual

            out_new = conv2(out)
            out_new = self.relu(out_new)
            out_new = bn2(out_new)
            out_new += out
                      
            if final_output is None:
                final_output = scale * out_new
            else:
                final_output += scale * out_new

        return final_output



class downsample_layer(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1, bias=False):
        super(downsample_layer, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=False)
        self.batch_norm = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return x



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        self.num_bases = 5
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True) #don't quantize the first layer
        self.bn1 = nn.BatchNorm2d(64) 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  #don't quantize the last layer
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = downsample_layer(self.inplanes, planes * block.expansion, 
                          kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(block(self.num_bases, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.num_bases, self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.avgpool(x4)
        x4 = x4.view(x4.size(0), -1)
        x5 = self.fc(x4)

        return {"preds": x5}


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        load_dict = torch.load('./full_precision_weights/model_best.pth.tar')['state_dict']
        model_dict = model.state_dict()
        model_keys = model_dict.keys()
        for name, param in load_dict.items():
            if name.replace('module.', '') in model_keys:
                model_dict[name.replace('module.', '')] = param    
        model.load_state_dict(model_dict)  
    return model


def resnet34(bitW, bitA, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], bitW, bitA, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(bitW, bitA, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], bitW, bitA, **kwargs)
    return model
