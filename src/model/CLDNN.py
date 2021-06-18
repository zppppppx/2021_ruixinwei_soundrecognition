import torch
import os
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer
from torch.utils.data import DataLoader, Dataset, dataset

import torchaudio
import pandas as pd
import numpy as np



class CLDNN(nn.Module):
    def __init__(self, mels=84, foresee: int=10, lookback: int=10):
        """
        Args:
            mels: the number of mels we want to train to get.
            foresee: how long we want to foresee.
            lookback: how long we want to look back.
        """
        super(CLDNN, self).__init__()
        self.mels = mels
        self.foresee = foresee
        self.lookback = lookback
        self.length = foresee + lookback + 1
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Time convolution
        self.Conv1 = nn.Conv1d(1, mels, 160)
        self.MP1 = nn.MaxPool1d(161)

        #  Frequency convolution
        self.Conv2 = nn.Conv2d(1, 64, (7, 13))
        self.MP2 = nn.MaxPool2d((3,6))

        # LSTM
        self.LSTM = nn.LSTM(input_size=64*12, hidden_size=128, num_layers=2,
                            batch_first=True, bidirectional=True)

        # DNN
        self.fc1 = nn.Linear(5*128*2, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 4)

        # Dropout
        self.Dropout1 = nn.Dropout(0.2)
        self.Dropout2 = nn.Dropout(0.35)

    def forward(self, inputs):
        # extract freq feature
        batchsize = inputs.shape[0]
        freq = torch.zeros((batchsize, self.length, self.mels)).to(self.device)

        for i in range(self.length):
            # print(inputs.shape)
            x_temp = inputs.reshape((batchsize, 1, self.length, 320))
            x = x_temp[:, 0, i, :].reshape(batchsize, 1, -1)
            # print(x.shape)
            x = self.Conv1(x)
            # print(x.shape)
            x = self.MP1(x)
            x = F.relu(x)
            x = torch.log(x+0.05)
            x = x.reshape(-1, self.mels)
            freq[:, i, :] = x

        freq = freq[:,None, :]
        
        # Frequency robustion
        x = self.Conv2(freq)
        x = self.MP2(x)

        # Relationship between before and after
        x = x.permute([0,2,1,3]).reshape((-1, 5, 64*12))
        x, (h, c) = self.LSTM(x)
        x = x.reshape(batchsize, -1)
        # print(x.shape)

        # DNN
        x = self.fc1(x)
        x = self.Dropout1(F.relu(x))
        x = self.fc2(x)
        x = self.Dropout2(F.relu(x))

        return self.fc3(x)


class raw_data(Dataset):
    def __init__(self, wav_path: str, label_file='../data/tr_piece.csv', root_path='../data/train/') -> None:
        self.root_path = root_path
        self.wav_path = root_path + wav_path
        self.label_file = label_file
        self.wav_name = wav_path.split('.')[0]
        self.data, self.labels = self.frag_load(self.wav_path)

    def frag_load(self, wav_path):
        frags = torch.zeros((89959, 1, 320*21))
        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform[0]

        for i in range(89959):
            # print(frags[i, 0].shape)
            frags[i, 0] = waveform[i*160:i*160+320*21]

        df = pd.read_csv(self.label_file)
        df = df['label_index'][df['id'] == self.wav_name]
        index = df.values[20:20+89959]
        index[index == 3] = 0
        index[index < 3] = 1

        return frags, torch.tensor(index).long()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

        
if __name__ == '__main__':
    # setting basic parameters
    train_path = '../data/train'
    model_cldnn = '../saved_models/cldnn.pkl'
    files = os.listdir(train_path)
    batch_size = 1024

    wav_val = '_7oWZq_s_Sk.wav'

    # set up the network
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss().to(devices)
    cldnn = CLDNN().to(devices)
    optimizer = optim.ASGD(cldnn.parameters(), lr=0.01, weight_decay=0.0000019156)
    # optimizer =  optim.Adam([p for p in cldnn.parameters() if p.requires_grad], lr=0.00018964, weight_decay=0.0000019156)

    for outer_epoch in range(2):
        i = 0
        for wav_path in files:
            if os.path.exists(model_cldnn):
                cldnn.load_state_dict(torch.load(model_cldnn))
            raw_dataset = raw_data(wav_path)
            raw_loader = DataLoader(dataset=raw_dataset, batch_size=batch_size, shuffle=True)

            # Training
            for inner_epoch in range(2):
                running_loss = 0.
                for idx, data in enumerate(raw_loader, 0):
                    inputs, labels = data[0].to(devices), data[1].to(devices)
                    # print(inputs.shape)
                    optimizer.zero_grad()
                    
                    outputs = cldnn(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    
                    if idx % 30 == 29:
                        print('[outer_epoch %d, file num %d, inner_epoch %d, batch_idx %d: loss %.5f'
                            %(outer_epoch, i, inner_epoch, idx, running_loss/30))
                        running_loss = 0.

            torch.save(cldnn.state_dict(), model_cldnn)

            i += 1

            valdata = raw_data(wav_val, label_file='../data/val_piece.csv', root_path='../data/val/')
            valloader = DataLoader(dataset=valdata, batch_size=batch_size)
            correct = 0
            total = 0
            with torch.no_grad():
                for data in valloader:
                    inputs, labels = data[0].to(devices), data[1].to(devices)
                    outputs = cldnn(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the all val inputs: %d %%' % (
                100 * correct / total))