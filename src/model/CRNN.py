import torch
import os
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio import transforms
import torchaudio
import pandas as pd
import numpy as np


# from cnn_evaluate import fill_and_extract
from model.CRNN_dataset import crnn_dataset
from model.CNN import *

def fill_and_extract(sound_file, frame_length=128, method='series', padding_mode='constant', saving=False, save_path = r'..\data\for_val\val.npy'):
    """
    In order to mark a piece of audio file, we consider padding the audio file at the beginning
    and the end, each time we select a piece with length of frame_length*128 (here frame_length
    denotes the smallest time span we want to classify, 128 denotes we need to grab 128 segments).

    Args:
        file_path: str, the audio file in need of extraction.
        frame_length: int, the smallest time span we want to classify.
        method: different extraction methods, 'series' for grabing a series to extract features containing features
            before and after the frame; 'sole' for grab a frame and padding it to demanded length.
        padding_mode: str, same as the np.pad, {'constant', 'symmetric', 'reflect'}
        saving: bool, denoting whether you want to save the file.

    Returns:
        features: tensor, size is [feature_tensor_nums, channel=1, segment_length, 128 bands]
    """
    wav_data, sample_rate = torchaudio.load(sound_file)
    tensor_number = int(len(wav_data[0])/160) # denotes the number of feature tensors

    features = torch.zeros((tensor_number, 1, 128, 40))
    timespan = int(127*frame_length/2)
    hop_length = int(frame_length/2)

    if method == 'series':
        head = 63*hop_length
        tail = 64*hop_length

        # fill the data in order to assure the length matches the number of frames
        wav_data = torch.tensor(np.pad(wav_data[0], (head, tail), mode=padding_mode)) 
        
        for i in range(tensor_number):
            start = i*160 #int(i*frame_length)
            mfcc = transforms.MFCC(n_mfcc=40, melkwargs={'n_mels':64, 'win_length':frame_length, 
            'hop_length':hop_length, 'n_fft':128})(wav_data[start:(start+timespan)])
            features[i, 0, :, :] = mfcc.T

            if i % 10000 == 0:
                print('{} pieces have been processed and {} are left.'.format(i, tensor_number-i))

    elif method == 'sole':
        for i in range(tensor_number):
            head = 62*hop_length
            tail = 63*hop_length
            wav_piece = wav_data[0][i*frame_length:(i+1)*frame_length]
            wav_for_val = torch.tensor(np.pad(wav_piece, (head, tail), mode='symmetric'))
            mfcc = transforms.MFCC(n_mfcc=40, melkwargs={'n_mels':64, 'win_length':frame_length, 
            'hop_length':hop_length, 'n_fft':128})(wav_for_val)
            
            features[i, 0, :, :] = mfcc.T

            if i % 10000 == 0:
                print('{} pieces have been processed and {} are left.'.format(i, tensor_number-i))

    if saving:
        # save_path = r'..\data\for_val\val.npy'
        np.save(save_path, features.numpy())

    print('Feature extraction finished!')

    return features

# class CRNN(nn.Module):
#     def __init__(self, classnums, hidden_size=256, num_layers=4, bidirectional=True):
#         super(CRNN, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.bi = 2 if bidirectional else 1

#         cnn = CNN(4)
#         model_cnn = '../saved_models/cnn.pkl'
#         cnn.load_state_dict(torch.load(model_cnn))
#         cnnlayers = list(cnn.children())
#         self.Conv1 = cnnlayers[0]
#         self.Conv2 = cnnlayers[1]
#         self.Conv3 = cnnlayers[2]
#         self.MP = cnnlayers[5]
#         self.Dropout1 = cnnlayers[6]
#         self.Dropout2 = cnnlayers[7]
#         self.Dropout3 = cnnlayers[8]

#         cnn_layers = [self.Conv1, self.Conv2, self.Conv3, self.MP, self.Dropout1, self.Dropout2, self.Dropout3]
#         # for layer in cnn_layers:
#         #     for layer_para in layer.parameters():
#         #         layer_para.requires_grad = False

#         self.LSTM = nn.LSTM(input_size=128*32, hidden_size=self.hidden_size, 
#                             num_layers=self.num_layers, batch_first=True, bidirectional=True)
#         self.FC1 = nn.Linear(8*self.bi*self.hidden_size, 64)
#         self.FC2 = nn.Linear(64, classnums)

#     def forward(self, x: tensor):
#         x = F.relu(self.Conv1(x))
#         x = self.MP(self.Dropout1(x))
#         x = F.relu(self.Conv2(x))
#         x = self.MP(self.Dropout2(x))
#         x = F.relu(self.Conv3(x))
#         x = self.Dropout3(x)

#         # print(x.shape)
#         x = x.permute([0,2,1,3]).reshape((-1, 8, 128*32))
#         x, (h, c) = self.LSTM(x)

#         # print(x.shape)
#         x = x.reshape((-1, 8*self.bi*self.hidden_size))
#         # print(x.shape)
#         x = F.relu(self.FC1(x))
#         # print(x.shape)
#         return self.FC2(x)


class CRNN2(nn.Module):
    def __init__(self, classnums, hidden_size=128, num_layers=4, bidirectional=True):
        super(CRNN2, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bi = 2 if bidirectional else 1

        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=(2,2), padding_mode='reflect')
        self.Conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=(1,1), padding_mode='reflect')
        self.Conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=(2,2), padding_mode='reflect')
        self.MP = nn.MaxPool2d(kernel_size=(4,2))
        self.Dropout1 = nn.Dropout(0.2)
        self.Dropout2 = nn.Dropout(0.3)
        self.Dropout3 = nn.Dropout(0.35)
        self.Dropout4 = nn.Dropout(0.2)

        self.LSTM = nn.LSTM(input_size=64*10, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.FC1 = nn.Linear(8*self.bi*self.hidden_size, 64)
        self.FC2 = nn.Linear(64, 32)
        self.FC3 = nn.Linear(32, classnums)

    def forward(self, x: tensor):
        x = F.relu(self.Conv1(x))
        x = self.MP(self.Dropout1(x))
        x = F.relu(self.Conv2(x))
        x = self.MP(self.Dropout2(x))
        x = F.relu(self.Conv3(x))
        x = self.Dropout3(x)

        # print(x.shape)
        x = x.permute([0,2,1,3]).reshape((-1, 8, 10*64))
        x, (h, c) = self.LSTM(x)

        # print(x.shape)
        x = x.reshape((-1, 8*self.bi*self.hidden_size))
        # print(x.shape)
        x = F.relu(self.FC1(x))
        # print(x.shape)
        x = F.relu(self.FC2(x))
        # print(x.shape)
        x = self.Dropout4(x)
        # print(x.shape)
        return self.FC3(x)


if __name__ == '__main__':
    # print(len(list(cnn.children())))
    root_path = '../data'
    model_cnn = '../saved_models/cnn.pkl'
    model_crnn = '../saved_models/crnn.pkl'
    frame_length = 80
    feature_file = '../data/crnn_feature_train.npy'
    label_file='../data/crnn_label_train.npy'
    batch_size = 128

    # Setting the network
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss().to(devices)
    crnn = CRNN2(classnums=2).to(devices)
    optimizer = optim.Adam([p for p in crnn.parameters() if p.requires_grad], lr=0.00018964, weight_decay=0.0000019156)
    

    # Training section
    train_dir = root_path + '/train/'
    train_csv = root_path + '/tr_piece.csv'
    files = os.listdir(train_dir)
    running_files = [*map(lambda x: train_dir + x, files)]
    # print(files)
    df = pd.read_csv(train_csv)

    for outter_epoch in range(2):
        for i in range(len(files[:])):
            if os.path.exists(model_crnn):
                crnn.load_state_dict(torch.load(model_crnn))

            sound_file = running_files[i]
            id = files[i].split('.')[0]
            labels = df['label_index'][df['id'] == id].values
            labels[labels<3] = 1
            labels[labels==3] = 0
            print(len(labels))
            if len(labels) < 90000:
                pad_head = int((90000-len(labels))/2)
                pad_tail = 90000 - len(labels) - pad_head
                labels = np.pad(labels, (pad_head,pad_tail), mode='symmetric')

            if len(labels) > 90000:
                start = int((len(labels)-90000)/2)
                labels = labels[start:start+90000]
            print(len(labels))
            np.save(label_file, labels)

            fill_and_extract(sound_file, frame_length, 'series', 'symmetric', True, save_path=feature_file)
            crnn_data = crnn_dataset()
            crnn_loader = DataLoader(dataset=crnn_data, batch_size=batch_size, shuffle=True)

            for inner_epoch in range(5):
                running_loss = 0.
                for idx, data in enumerate(crnn_loader, 0):
                    inputs, labels = data[0].to(devices), data[1].to(devices)
                    
                    optimizer.zero_grad()
                    
                    outputs = crnn(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    
                    if idx % 100 == 99:
                        print('[outer_epoch %d, file num %d, inner_epoch %d, batch_idx %d: loss %.5f'
                            %(outter_epoch, i, inner_epoch, idx, running_loss/100))
                        running_loss = 0.

            torch.save(crnn.state_dict(), model_crnn)





    