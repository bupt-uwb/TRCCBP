import torch.nn as nn
from .blocks import *
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_block1 = CNN(kernel_size=3)
        self.BN = nn.BatchNorm1d(num_features=32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.drop = nn.Dropout(0.2)
        self.conv_block2 = CNN(kernel_size=5)
        self.gru = GRU()
        self.multihead_attention = MultiHead_Attention(in_ch=128, num_head=8)
        self.fc1 = FC(28*128, 100)
        self.fc2 = FC(100, 2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x)
        x3 = torch.cat([x1,x2],dim=1)
        # print(x3.shape)
        x3 = x3.transpose(1, 2)
        # print(x3.shape)
        x3 = self.gru(x3)
        x3 = self.multihead_attention(x3)
        # x3 = self.act(x3)
        x3 = x3.transpose(1, 2)
        # print(x.shape)
        x3 = self.fc1(x3)
        x3 = self.act(x3)
        # # print(x.shape)
        out = self.fc2(x3)
        return out


if __name__ == '__main__':
    data1 = torch.randn([32, 1, 30])
    # print(type(data1))
    # data2 = torch.randn([32, 50, 1, 3])
    model = Net()
    out = model(data1)
    print(out.shape)
