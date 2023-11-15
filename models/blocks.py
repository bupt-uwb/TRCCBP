import torch.nn as nn
import torch
from torch.nn.utils import weight_norm


class CNN(nn.Module):
    def __init__(self, kernel_size):
        super(CNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=kernel_size,padding=(kernel_size-1)//2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2,stride=1),
            # nn.Dropout(0.2),
            # nn.Conv1d(32, 32, kernel_size=kernel_size,padding=(kernel_size-1)//2),
            # nn.BatchNorm1d(32),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(32, 32, kernel_size=kernel_size,padding=(kernel_size-1)//2),
            # nn.BatchNorm1d(32),
            # nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2,stride=1),
            # nn.Dropout(0.2)
        )

    def forward(self, x):
        output = self.conv_block(x)
        # print(output.size())
        return output

class GRU(nn.Module):

    def __init__(self):
        super().__init__()
        self.GRU = nn.GRU(input_size=128, hidden_size=128, num_layers=1, bidirectional=False,
                             batch_first=True)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x, h_n = self.GRU(x)
        return x


class FC(nn.Module):
    def __init__(self,input, output):
        super(FC, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input, output)
        )

    def forward(self, x):
        output = self.fc_layer(x)
        # print(output.size())
        return output

class MultiHead_Attention(nn.Module):
    def __init__(self, in_ch, num_head):
        '''
        Args:
            dim: dimension for each time step
            num_head:num head for multi-head self-attention
        '''
        super().__init__()
        self.dim = in_ch
        self.num_head = num_head
        self.qkv = nn.Linear(in_ch, in_ch * 3)  # extend the dimension for later spliting

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = q @ k.transpose(-1, -2)
        att = att.softmax(dim=1)  # 将多个注意力矩阵合并为一个
        x = (att @ v).transpose(1, 2)
        x = x.reshape(B, N, C)
        return x


if __name__ == '__main__':
    data = torch.randn([32, 1, 30])
    # print(data.shape[0])
    # data2 = torch.randn([64, 50, 1, 640]).reshape([64*50,1,640])
    # data3 = data2.reshape([64,50,1,640])
    # model = UPCNN(32,16)
    model = CNN(5)
    # model = CNN()
    out = model(data)
    # out = out[:,:,9:-8]
    print(out.shape)
    # print(data3.shape)
