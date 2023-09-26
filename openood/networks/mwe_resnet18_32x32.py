import torch
import torch.nn as nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
from torch import Tensor
import torch.nn.functional as F


class CustomLinear(torch.nn.Linear):
    
    def __init__(self, in_features: int,
                 out_features: int,
                 bias: bool = True) -> None:
        
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias)
        
        self.weight_noise = torch.nn.Parameter(torch.ones_like(self.weight)*0.0001,
                                               requires_grad = True)
        self.weight.requires_grad = False
        if bias:
            self.bias_noise = torch.nn.Parameter(torch.ones_like(self.bias)*0.0001,
                                                  requires_grad = True)
            self.bias.requires_grad = False
            self.bias_is_not_None_ = True
        else:
            self.bias_noise = None
            self.bias_is_not_None_ = False
            

    def forward(self, x: Tensor) -> Tensor:
        z = torch.rand_like(self.weight_noise).cuda()*2.-1.
        weight = self.weight + z * self.weight_noise
        
        if self.bias_is_not_None_:
            z = torch.rand_like(self.bias_noise).cuda()*2.-1.
            bias = self.bias + z * self.bias_noise
        else:
            bias = self.bias
        
        return F.linear(x, weight, bias)
    
    
    def load_state_dict(self, state_dict):
        if "weight_noise" not in state_dict:
            state_dict["weight_noise"] = self.weight_noise
        if "bias_noise" not in state_dict and self.bias_is_not_None_:
            state_dict["bias_noise"] = self.bias_noise
        super().load_state_dict(state_dict)
        

class CustomConv2d(torch.nn.Conv2d):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ) -> None:
        
        super().__init__(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode)
        
        self.weight_noise = torch.nn.Parameter(torch.ones_like(self.weight)*0.0001,
                                               requires_grad = True)
        self.weight.requires_grad = False
        if bias:
            self.bias_noise = torch.nn.Parameter(torch.ones_like(self.bias)*0.0001,
                                                  requires_grad = True)
            self.bias.requires_grad = False
            self.bias_is_not_None_ = True
        else:
            self.bias_noise = None
            self.bias_is_not_None_ = False
            

    def forward(self, x: Tensor) -> Tensor:
        z = torch.rand_like(self.weight_noise).cuda()*2.-1.
        weight = self.weight + z * self.weight_noise
        
        if self.bias_is_not_None_:
            z = torch.rand_like(self.bias_noise).cuda()*2.-1.
            bias = self.bias + z * self.bias_noise
        else:
            bias = self.bias
        
        return self._conv_forward(x, weight, bias)
    
    
    def load_state_dict(self, state_dict):
        if "weight_noise" not in state_dict:
            state_dict["weight_noise"] = self.weight_noise
        if "bias_noise" not in state_dict and self.bias_is_not_None_:
            state_dict["bias_noise"] = self.bias_noise
        super().load_state_dict(state_dict)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = CustomConv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = CustomConv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                CustomConv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = CustomConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = CustomConv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = CustomConv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                CustomConv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MWE_ResNet18_32x32(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=None, num_classes=10):
        super(MWE_ResNet18_32x32, self).__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        self.in_planes = 64

        self.conv1 = CustomConv2d(3,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.avgpool = nn.AvgPool2d(4)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = CustomLinear(512 * block.expansion, num_classes)
        self.feature_size = 512 * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False, return_feature_list=False):
        feature1 = F.relu(self.bn1(self.conv1(x)))
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        logits_cls = self.fc(feature)
        feature_list = [feature1, feature2, feature3, feature4, feature5]
        
        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls

    def forward_threshold(self, x, threshold):
        feature1 = F.relu(self.bn1(self.conv1(x)))
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.fc(feature)

        return logits_cls

    def get_fc(self):
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()
