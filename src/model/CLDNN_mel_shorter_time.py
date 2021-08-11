import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer
from torch.utils.data import DataLoader, Dataset

import torchaudio
import torchaudio.transforms as transforms
import pandas as pd
import numpy as np
import random



'''class CLDNN(nn.Module):
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

        # BatchNorm
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(1)
        self.bn3 = nn.BatchNorm1d(1)

        # Time convolution
        self.Conv1 = nn.Conv1d(1, mels, 160)
        self.MP1 = nn.MaxPool1d(161)

        #  Frequency convolution
        self.Conv2 = nn.Conv2d(1, 64, (1, 13))
        self.MP2 = nn.MaxPool2d((1,6))

        # LSTM
        self.LSTM1 = nn.LSTM(input_size=64*12, hidden_size=128, num_layers=3,
                            batch_first=True, bidirectional=True)

        self.LSTM2 = nn.LSTM(input_size=320, hidden_size=128, num_layers=2,
                            batch_first=True, bidirectional=True)

        # DNN
        self.fc1 = nn.Linear(21*128*2*2, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn. Linear(32, 2)

        self.fc4 = nn.Linear(112, 32)
        self.fc5 = nn. Linear(32, 2)

        # Last Conv1d
        self.Conv3 = nn.Conv1d(2, 8, 128, 128, 64)
        self.MP3 = nn.MaxPool1d(3)

        # Dropout
        self.Dropout1 = nn.Dropout(0.2)
        self.Dropout2 = nn.Dropout(0.35)

    def forward(self, inputs):
        time_inputs = self.bn1(inputs)
        freq_inputs = inputs
        time_inputs = time_inputs.reshape(-1, 21, 320)


        # extract freq feature and 
        batchsize = inputs.shape[0]
        # print(batchsize)
        freq = torch.zeros((batchsize, self.length, self.mels)).to(self.device)

        for i in range(self.length):
            # print(inputs.shape)
            x_temp = freq_inputs.reshape((batchsize, 1, self.length, 320))
            x = x_temp[:, 0, i, :].reshape(batchsize, 1, -1)
            # print(x.shape)
            x = self.Conv1(x)
            # print(x.shape)
            x = self.MP1(x)
            x = F.relu(x)
            # print('x=', x)
            x = torch.log(x+0.00001)
            x = x.reshape(-1, self.mels)
            freq[:, i, :] = x

        freq = freq[:,None, :]
        
        # Frequency robustion
        x = self.Conv2(freq)
        x = self.MP2(x)

        # Relationship between before and after
        x = x.permute([0,2,1,3]).reshape((-1, 21, 64*12))
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

        # two_feature = torch.cat((x, time), dim=1)
        two_feature = torch.cat((x[:, None, :], time[:, None, :]), dim=1)
        # print(two_feature.shape)

        # Conv1d
        out = self.Conv3(two_feature)
        out = self.MP3(out)
        out = out.reshape((-1, 112))

        # DNN
        x = self.fc4(out)
        x = self.Dropout1(F.relu(x))
        # x = self.fc2(x)
        # x = self.Dropout2(F.relu(x))

        return self.fc5(x)

'''

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
        # self.MP2 = nn.MaxPool2d((2,2))

        # LSTM
        self.LSTM1 = nn.LSTM(input_size=128*14, hidden_size=128, num_layers=3,
                            batch_first=True, bidirectional=True) # Frequency

        self.LSTM2 = nn.LSTM(input_size=320, hidden_size=128, num_layers=2,
                            batch_first=True, bidirectional=True) # Time

        # DNN
        self.fc1 = nn.Linear(21*128*2+14*128*2, 512)
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
        mel_spec = transforms.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=320, hop_length=192, n_mels=36).to(self.device)
        freq = (mel_spec(freq_inputs)+0.00001).log2()
        
        # Frequency Conv
        x = self.Conv1(freq)
        x = F.relu(x)
        x = self.Dropout0(x)
        # x = self.MP1(x)
        x = self.Conv2(x)
        x = self.MP1(x)

        # Relationship between before and after
        x = x.permute([0,3,1,2]).reshape((batchsize, 14, -1))
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

        return F.softmax(self.fc3(x), dim=1)


class raw_data(Dataset):
    def __init__(self, wav_path: str, label_file='../data/tr_piece.csv', root_path='../data/train/') -> None:
        self.root_path = root_path
        self.wav_path = wav_path
        self.label_file = label_file
        # self.wav_name = wav_path.split('.')[0]
        self.data, self.labels = self.frag_load(self.wav_path)

    def frag_load(self, wav_path):
        data = torch.tensor([])
        labels = torch.tensor([])
        size = 8000
        for wav_file in wav_path:
            wav_name = wav_file.split('.')[0]
            wav_file = self.root_path + wav_file
            frags = torch.zeros((size, 1, 320*21))
            waveform, sr = torchaudio.load(wav_file)

            waveform = waveform[0]
            num = np.random.randint(89948, size=size) # randomly pick up some pieces
            for i in range(size):
                # print(frags[i, 0].shape)
                index = num[i]
                frags[i, 0] = waveform[index*160:index*160+320*21]

            df = pd.read_csv(self.label_file)
            df = df['label_index'][df['id'] == wav_name]
            index = df.values[20:20+89959]

            # index0_2 = index[index < 3]
            # index3 = index[index == 3][:len(index0_2)]
            # # print(index3[:10], index0_2[:10])
            # frags3 = frags[index == 3][:len(index0_2)]
            # frags0_2 = frags[index < 3]
            # # print(len(index3), len(index0_2))
            # frags = np.concatenate((frags3, frags0_2))
            # index = np.concatenate((index3, index0_2))
            index[index<3] = 1
            index[index==3] = 0

            index = index[num]
            index = torch.from_numpy(index)
            data = torch.cat((data, frags), dim=0)
            labels = torch.cat((labels, index), dim=0)

        return data.clone().detach().float(), labels.clone().detach().long()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

        
if __name__ == '__main__':
    # setting basic parameters
    train_path = '../data/train'
    val_path = '../data/val'
    model_cldnn = '../saved_models/cldnn_mel_log2_s.pkl'
    files = os.listdir(train_path)
    batch_size = 128


    wav_val = os.listdir(val_path) #['J1jDc2rTJlg.wav','Kb1fduj-jdY.wav','t1LXrJOvPDg.wav', 'yMtGmGa8KZ0.wav']


    # set up the network
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss().to(devices)
    cldnn = CLDNN_mel().to(devices)
    # optimizer = optim.ASGD(cldnn.parameters(), lr=0.01, weight_decay=0.0000019156)
    optimizer =  optim.Adam([p for p in cldnn.parameters() if p.requires_grad], lr=0.00018964, weight_decay=0.0000019156)

    # for outer_epoch in range(2):
    #     i = 0

    # for wav_path in files:
    for j in range(100):
        wav_path = random.sample(files, 30) # randomly pick some files as the input

        if os.path.exists(model_cldnn): 
            cldnn.load_state_dict(torch.load(model_cldnn))
        raw_dataset = raw_data(wav_path)
        raw_loader = DataLoader(dataset=raw_dataset, batch_size=batch_size, shuffle=True)

        # Training
        for inner_epoch in range(1):
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
                
                if idx % 100 == 99:
                    print('[pick order %d, inner_epoch %d, batch_idx %d: loss %.5f'
                        %(j, inner_epoch, idx, running_loss/100))
                    running_loss = 0.

        
        raw_dataset, raw_loader = None, None
        # i += 1

# cldnn = CLDNN().to(devices)
# cldnn.load_state_dict(torch.load(model_cldnn))

        if j % 2 == 1:
            valdata = raw_data(wav_val, label_file='../data/val_piece.csv', root_path='../data/val/')
            valloader = DataLoader(dataset=valdata, batch_size=batch_size)
            correct = 0
            total = 0
            with torch.no_grad():
                for data in valloader:
                    inputs, labels = data[0].to(devices), data[1].to(devices)
                    # print(labels[labels==1].shape)
                    # print(labels[labels==0].shape)
                    # print(labels)
                    outputs = cldnn(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the all val inputs: %d %%' % (
                100 * correct / total))
            valdata, valloader = None, None
            
        torch.save(cldnn.state_dict(), model_cldnn)