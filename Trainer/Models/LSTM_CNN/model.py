import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights
from torchinfo import summary
import math

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size=2097152, hidden_size=64, num_layers=1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 1)
       
    def forward(self, x_3d):
        hidden = None
        # Iterate over each frame of a video in a video of batch * frames * channels * height * width
        for t in range(x_3d.size(1)):
            x = x_3d[:,t]
            x = self.conv1(x)
            x = self.act1(x)
            x = self.drop1(x)
            x = self.conv2(x)
            x = self.act2(x)
            x = self.pool2(x)
            x = self.flatten(x)
            out, hidden = self.lstm(x, hidden)       

        # Get the last hidden state (hidden is a tuple with both hidden and cell state in it)
        x = self.fc1(hidden[0][-1])
        x = F.relu(x)
        x = self.fc2(x)
        return x, F.sigmoid(x)

if __name__ == '__main__':
    model = CNNLSTM().cuda()
    summary(model,(8,10,3,512,512))