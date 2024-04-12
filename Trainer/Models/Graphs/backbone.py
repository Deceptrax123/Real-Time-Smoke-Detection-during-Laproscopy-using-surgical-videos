import torch
from torch.nn import Module, Conv3d, MaxPool3d, BatchNorm3d, ReLU, Linear
import torch.nn.functional as f
from torchsummary import summary


class Conv3DBase(Module):
    def __init__(self):
        super(Conv3DBase, self).__init__()

        self.conv1 = Conv3d(in_channels=3, out_channels=4,
                            stride=1, padding=1, kernel_size=(3, 3, 3))
        self.bn1 = BatchNorm3d(4)
        self.relu1 = ReLU()
        self.pool1 = MaxPool3d((3, 3, 3), stride=(1, 2, 2), padding=1)

        self.conv2 = Conv3d(in_channels=4, out_channels=8,
                            stride=1, padding=1, kernel_size=(3, 3, 3))
        self.bn2 = BatchNorm3d(8)
        self.relu2 = ReLU()
        self.pool2 = MaxPool3d((3, 3, 3), stride=(1, 2, 2), padding=1)

        self.conv3 = Conv3d(in_channels=8, out_channels=16,
                            stride=1, padding=1, kernel_size=(3, 3, 3))
        self.bn3 = BatchNorm3d(16)
        self.relu3 = ReLU()
        self.pool3 = MaxPool3d((3, 3, 3), stride=(1, 2, 2), padding=1)

        self.conv4 = Conv3d(in_channels=16, out_channels=32,
                            stride=1, padding=1, kernel_size=(3, 3, 3))
        self.bn4 = BatchNorm3d(32)
        self.relu4 = ReLU()
        self.pool4 = MaxPool3d((3, 3, 3), stride=(1, 2, 2), padding=1)

        self.conv5 = Conv3d(in_channels=32, out_channels=64,
                            stride=1, padding=1, kernel_size=(3, 3, 3))
        self.bn5 = BatchNorm3d(64)
        self.relu5 = ReLU()
        self.pool5 = MaxPool3d((3, 3, 3), stride=(1, 2, 2), padding=1)

        self.conv6 = Conv3d(in_channels=64, out_channels=128,
                            stride=1, padding=1, kernel_size=(3, 3, 3))
        self.bn6 = BatchNorm3d(128)
        self.relu6 = ReLU()
        self.pool6 = MaxPool3d((3, 3, 3), stride=(1, 2, 2), padding=1)

        self.conv7 = Conv3d(in_channels=128, out_channels=128,
                            stride=1, padding=1, kernel_size=(3, 3, 3))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.pool6(x)

        x = self.conv7(x)

        x = torch.view(x.size(0), x.size(1)*x.size(2)*x.size(3)*x.size(4))

        return x
