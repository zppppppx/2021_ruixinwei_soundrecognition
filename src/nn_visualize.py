import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torchaudio
import torchaudio.transforms as transforms
import pandas as pd
import numpy as np
import torch.nn.functional as F
import random

class CLDNN_mel(nn.Module):
    def __init__(self, mels=84, foresee: int=10, lookback: int=10):
        """
        Args:
            mels: the number of mels we want to train to get.
            foresee: how long we want to foresee.
            lookback: how long we want to look back.
        """
        super(CLDNN_mel, self).__init__()
        self.mels = mels
        self.foresee = foresee
        self.lookback = lookback
        self.length = foresee + lookback + 1
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # BatchNorm
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(1)
        self.bn3 = nn.BatchNorm1d(1)

        # Time convolution
        self.Conv1 = nn.Conv2d(1, 32, 5)
        self.MP1 = nn.MaxPool2d((2,2))

        #  Frequency convolution
        self.Conv2 = nn.Conv2d(32, 128, 5)
        self.MP2 = nn.MaxPool2d((2,2))

        # LSTM
        self.LSTM1 = nn.LSTM(input_size=128*6, hidden_size=128, num_layers=3,
                            batch_first=True, bidirectional=True)

        self.LSTM2 = nn.LSTM(input_size=320, hidden_size=128, num_layers=2,
                            batch_first=True, bidirectional=True)

        # DNN
        self.fc1 = nn.Linear(21*128*2+6*128*2, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn. Linear(32, 2)

        # Dropout
        self.Dropout0 = nn.Dropout(0.1)
        self.Dropout1 = nn.Dropout(0.2)
        self.Dropout2 = nn.Dropout(0.35)

    def forward(self, inputs):
        time_inputs = self.bn1(inputs)
        freq_inputs = inputs
        time_inputs = time_inputs.reshape(-1, 21, 320)


        # extract freq feature and 
        batchsize = inputs.shape[0]
        # print(batchsize)
        mel_spec = transforms.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=320, hop_length=192, n_mels=36)
        freq = mel_spec(freq_inputs)
        
        # Frequency Conv
        x = self.Conv1(freq)
        x = self.Dropout0(x)
        x = self.MP1(x)
        x = self.Conv2(x)
        x = self.MP2(x)

        # Relationship between before and after
        x = x.permute([0,3,1,2]).reshape((batchsize, 6, -1))
        x, (h, c) = self.LSTM1(x)
        # x = x.reshape(batchsize, 1, -1)
        # x = self.bn2(x)
        x = x.reshape(batchsize, -1)

        # print(x.shape)

        # Time feature
        time, (h, c) = self.LSTM2(time_inputs)
        # time = time.reshape(batchsize, 1, -1)
        # time = self.bn3(time)
        time = time.reshape(batchsize, -1)

        two_feature = torch.cat((x, time), dim=1)
        # print(two_feature.shape)

        # DNN
        x = self.fc1(two_feature)
        x = self.Dropout2(F.relu(x))
        x = self.fc2(x)
        # x = self.Dropout1(F.relu(x))
        x = F.relu(x)
        x = F.softmax(x, dim=1)

        return self.fc3(x)
 
dummy_input = torch.rand(100, 1, 6720) #假设输入13张1*28*28的图片
model = CLDNN_mel()
with SummaryWriter(comment='LeNet') as w:
    w.add_graph(model, (dummy_input, ))
