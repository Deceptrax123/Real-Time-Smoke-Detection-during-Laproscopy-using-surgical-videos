import torch
from torch import nn


def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if module.bias.data is not None:
                module.bias.data.zero_()
            else:
                nn.init.kaiming_normal_(
                    module.weight.data, mode='fan_in', nonlinearity='relu')
