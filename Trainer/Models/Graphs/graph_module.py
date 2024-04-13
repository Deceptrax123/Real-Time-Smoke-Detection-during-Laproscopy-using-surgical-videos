from torch import nn
import torch
from torch.nn import ReLU, Linear, Module, Sigmoid, Conv3d
from torch.nn.functional import softmax, sigmoid


class GraphConstructor(Module):
    def __init__(self, conv3dbase):
        super(GraphConstructor, self).__init__()

        self.extractor = conv3dbase  # Feature extractor based on 3D convolutio

        self.linear = Linear(in_features=128*10*8*8, out_features=320)

        # Spatial Graph Modules
        self.branch1 = Conv3d(in_channels=32, out_channels=16,
                              kernel_size=(1, 1, 1), stride=1)
        self.branch2 = Conv3d(in_channels=32, out_channels=16,
                              kernel_size=(1, 1, 1), stride=1)

        self.relu1 = ReLU()
        self.relu2 = ReLU()

        # Graph Convolution Layers
        self.g_1 = GraphConvolution(f=32)
        self.g_2 = GraphConvolution(f=32)
        self.g_3 = GraphConvolution(f=32)

        # classifier
        self.classifier = Linear(in_features=32*10, out_features=1)

    def forward(self, x):
        # Extract Features
        x = self.extractor.forward(x)

        # Make the Feature Extractor for 32 fetures
        x = self.linear(x)

        # Resize
        x = x.view(x.size(0), 32, 10, 1, 1)

        x1 = self.branch1(x)
        x1 = self.relu1(x1)

        x2 = self.branch2(x)
        x2 = self.relu2(x2)

        # Adjacency Matrix on spatial features
        x1 = x1.view(x1.size(0), x1.size(1), x1.size(2))
        x2 = x2.view(x2.size(0), x2.size(1), x2.size(2))

        A1 = softmax(torch.bmm(torch.transpose(x1, 1, 2), x2))

        # Adjacency Matrix on temporal features
        A2 = torch.zeros((x.size(0), 10, 10)).to(device="cuda")
        for k in range(x.size(0)):
            for i in range(10):  # 10 frames
                for j in range(10):  # 10 frames
                    A2[k][i][j] = torch.exp(-torch.abs(torch.tensor(i-j)))

        # Fuse Both Matrices
        A = torch.add(A1, A2)

        # Graph Convolution
        x_g = self.g_1(x, A)
        x_g = self.g_2(x_g, A)
        x_g = self.g_3(x_g, A)

        # Classifier
        x_g = x_g.view(x_g.size(0), x_g.size(1)*x_g.size(2)*x_g.size(3))
        x_g = self.classifier(x_g)

        return x_g, sigmoid(x_g)


class GraphConvolution(Module):
    def __init__(self, f):
        super(GraphConvolution, self).__init__()

        self.conv = Conv3d(in_channels=f, out_channels=f,
                           kernel_size=(1, 1, 1), stride=1)

    def forward(self, X, A):
        X = X.view(X.size(0), X.size(1), X.size(2))

        X_l = torch.bmm(X, A)
        X_l = X_l.view(X_l.size(0), X_l.size(1), X_l.size(2), 1, 1)
        X_l = self.conv(X_l)

        # Skip Connection
        X = X.view(X.size(0), X.size(1), X.size(2), 1, 1)
        X_l = X_l+X

        return X_l
