from Trainer.Models.Graphs.backbone import Conv3DBase
from torch import nn
import torch
from torch.nn import ReLU, Linear, Module
from torch.nn.functional import softmax


class GraphConstructor(Module):
    def __init__(self):
        super(GraphConstructor, self).__init__()

        self.extractor = Conv3DBase()  # Feature extractor based on 3D convolution

        # Spatial Graph Modules
        self.branch1 = Linear(in_features=128*10, out_features=10)
        self.branch2 = Linear(in_features=128*10, out_features=10)

        self.relu1 = ReLU()
        self.relu2 = ReLU()

    def forward(self, x):
        # Extract Features
        x = self.extractor(x)

        x1 = self.branch1(x)
        x1 = self.relu1(x)

        x2 = self.branch2(x)
        x2 = self.relu2(x)

        # Adjacency Matrix Creation
        A1 = softmax(x1*x2)
