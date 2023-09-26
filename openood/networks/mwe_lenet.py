import logging

import torch
import torch.nn as nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
from torch import Tensor
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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



class MWE_LeNet(nn.Module):
    def __init__(self, num_classes, num_channel=3):
        super(MWE_LeNet, self).__init__()
        self.num_classes = num_classes
        self.feature_size = 84
        self.block1 = nn.Sequential(
            CustomConv2d(in_channels=num_channel,
                      out_channels=6,
                      kernel_size=5,
                      stride=1,
                      padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))

        self.block2 = nn.Sequential(
            CustomConv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2))

        self.block3 = nn.Sequential(
            CustomConv2d(in_channels=16,
                      out_channels=120,
                      kernel_size=5,
                      stride=1), nn.ReLU())

        self.classifier1 = CustomLinear(in_features=120, out_features=84)
        self.relu = nn.ReLU()
        self.fc = CustomLinear(in_features=84, out_features=num_classes)

    def get_fc(self):
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def forward(self, x, return_feature=False, return_feature_list=False):
        feature1 = self.block1(x)
        feature2 = self.block2(feature1)
        feature3 = self.block3(feature2)
        feature3 = feature3.view(feature3.shape[0], -1)
        feature = self.relu(self.classifier1(feature3))
        logits_cls = self.fc(feature)
        feature_list = [feature1, feature2, feature3, feature]
        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls

    def forward_threshold(self, x, threshold):
        feature1 = self.block1(x)
        feature2 = self.block2(feature1)
        feature3 = self.block3(feature2)
        feature3 = feature3.view(feature3.shape[0], -1)
        feature = self.relu(self.classifier1(feature3))
        feature = feature.clip(max=threshold)
        logits_cls = self.fc(feature)

        return logits_cls
