import torch.nn as nn
from torch import Tensor
import torch
from torchvision.models import *
from torchsummary import summary
class MGSModel(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.net = efficientnet_v2_s(num_classes=34, dropout=dropout_rate)
    def forward(self, x, get_features=False):
        if get_features:
            features = self.net.features(x)
            features = self.net.avgpool(features)
            features = torch.flatten(features, 1)
            return features
        else:
            x = self.net(x)
            x = torch.sigmoid(x) * 100. + 1.
            return x, None
