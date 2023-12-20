import torch
from torch import nn
import torch.nn.functional as F
from cbam import *
import numpy as np

class CNN_LSTM_Attention(nn.Module):
    def __init__(self,):
        super(CNN_LSTM_Attention, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 9, 5)
        self.conv3 = nn.Conv2d(9, 6, 5)
        self.conv4 = nn.Conv2d(6, 3, 5)
        self.fc1 = nn.Linear(300, 125)
        self.fc2 = nn.Linear(225, 84)
        self.fc3 = nn.Linear(84,1)
        self.fc4 = nn.Linear(100, 84)
        self.cbam = CBAM(3)

        self.lstm = nn.LSTM(6, 100, num_layers=3, batch_first=True)

        # 定义self_attention 网络结构
        self.query_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        self.key_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        self.value_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):

        """"X为图片数据"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        attention_map = x

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))

        """Y为多维度数据"""
        y, _ = self.lstm(y)
        """合并"""
        outputs = torch.cat([x, y], dim=1)

        """Y为多维度数据"""

        """注意力机制"""
        outputs = outputs.reshape(outputs.shape[0], 1, 15, 15)
        # outputs = x.reshape(3, 1, 10, 10)
        m_batchsize, C, width, height = outputs.size()
        proj_query = self.query_conv(outputs).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(outputs).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(outputs).view(m_batchsize, -1, width * height)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # print("outputs.shape:",outputs.shape)
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + outputs
        # attention_map = out
        out = out.view(1, -1)
        # out = F.relu(self.fc1(out))
        x_platy = F.relu(self.fc2(out))
        x_lstm = F.relu(self.fc4(y))
        x_fix = F.relu(self.fc2(outputs.view(1, -1)))
        x_platy = self.fc3(x_platy)
        x_lstm = self.fc3(x_lstm)
        x_fix = self.fc3(x_fix)
        return x_platy, attention_map