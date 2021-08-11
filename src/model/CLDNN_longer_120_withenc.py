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
from model.encoder import enc_dec


class CLDNN_mel(nn.Module):
    def __init__(self, model_enc='../saved_models/enc_dec.pkl'):
        """
        Args:
            mels: the number of mels we want to train to get.
            foresee: how long we want to foresee.
            lookback: how long we want to look back.
        """
        super(CLDNN_mel, self).__init__()
    
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # BatchNorm
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(1)
        self.bn3 = nn.BatchNorm1d(1)

        # Frequecy convolution
        self.Conv1 = nn.Conv2d(1, 32, 5, padding=(2,2), padding_mode='reflect')
        self.MP1 = nn.MaxPool2d((2,3))
        self.Conv2 = nn.Conv2d(32, 128, 5, padding=(2,2), padding_mode='reflect')

        # LSTM
        self.LSTM1 = nn.LSTM(input_size=128*32, hidden_size=128, num_layers=3,
                            batch_first=True, bidirectional=True) # Frequency

        self.LSTM2 = nn.LSTM(input_size=320, hidden_size=64, num_layers=2,
                            batch_first=True, bidirectional=True) # Time

        # DNN
        self.fc1 = nn.Linear(61*64*2+27*128*2, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn. Linear(32, 2)

        # Dropout
        self.Dropout0 = nn.Dropout(0.1)
        self.Dropout1 = nn.Dropout(0.2)
        self.Dropout2 = nn.Dropout(0.35)

        # Encoder
        self.enc_dec_net = enc_dec()
        self.enc_dec_net.load_state_dict(torch.load(model_enc))
        for p in self.enc_dec_net.parameters():
            p.requires_grad = False

    def forward(self, inputs):
        time_inputs = self.bn1(inputs)
        time_inputs = time_inputs.reshape(-1, 61, 320)


        # extract freq feature
        mel_spec = transforms.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=320, hop_length=244, n_mels=64).to(self.device)
        freq = (mel_spec(inputs)+0.000001).log2() # to avoid NaN of loss
        inputs = None
        
        # Frequency Conv
        x = self.Conv1(freq)
        x = F.relu(x)
        x = self.Dropout0(x)
        x = self.MP1(x)
        x = self.Conv2(x)
        x = F.relu(x)

        # Relationship between before and after
        x = x.permute([0,3,1,2]).reshape((-1, 27, 128*32))
        x, (h, c) = self.LSTM1(x)
        h, c = None, None
        x = x.reshape(-1, 27*2*128)


        # Time feature
        time, (h, c) = self.LSTM2(time_inputs)
        h, c, time_inputs = None, None, None
        time = time.reshape(-1, 61*64*2)

        two_feature = torch.cat((x, time), dim=1) # cat


        # DNN
        x = self.fc1(two_feature)
        x = self.Dropout2(F.relu(x))
        x = self.fc2(x)
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
        size = 6000
        for wav_file in wav_path:
            wav_name = wav_file.split('.')[0]
            wav_file = self.root_path + wav_file
            frags = torch.zeros((size, 1, 320*61))
            waveform, sr = torchaudio.load(wav_file)

            waveform = waveform[0]
            num = np.random.randint(89800, size=size) # randomly pick up some pieces
            for i in range(size):
                # print(frags[i, 0].shape)
                index = num[i]
                frags[i, 0] = waveform[index*160:index*160+320*61]

            df = pd.read_csv(self.label_file)
            df = df['label_index'][df['id'] == wav_name]
            index = df.values[61:61+89800]

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
    model_cldnn = '../saved_models/cldnn_mel_longer_120_with_enc.pkl'
    model_enc = '../saved_models/enc_dec.pkl'
    files = os.listdir(train_path)
    batch_size = 128


    wav_val = os.listdir(val_path) #['J1jDc2rTJlg.wav','Kb1fduj-jdY.wav','t1LXrJOvPDg.wav', 'yMtGmGa8KZ0.wav']


    # set up the network
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss().to(devices)
    cldnn = CLDNN_mel().to(devices)
    optimizer =  optim.Adam([p for p in cldnn.parameters() if p.requires_grad], lr=0.00018964, weight_decay=0.0000019156)



    for j in range(100):
        wav_path = random.sample(files, 20) # randomly pick some files as the input

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


        if j % 2 == 1:
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
            valdata, valloader = None, None
            
        torch.save(cldnn.state_dict(), model_cldnn)