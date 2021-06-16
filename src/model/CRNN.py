import torch
import os
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio import transforms
import pandas as pd
import numpy as np


# from cnn_evaluate import fill_and_extract
from CRNN_dataset import crnn_dataset
from CNN import *

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
    tensor_number = int(len(wav_data[0])/frame_length) # denotes the number of feature tensors

    features = torch.zeros((tensor_number, 1, 128, 128))
    timespan = int(127*frame_length/2)
    hop_length = int(frame_length/2)

    if method == 'series':
        head = 63*hop_length
        tail = 64*hop_length

        # fill the data in order to assure the length matches the number of frames
        wav_data = torch.tensor(np.pad(wav_data[0], (head, tail), mode=padding_mode)) 
        
        for i in range(tensor_number):
            start = int(i*frame_length)
            mfcc = transforms.MFCC(n_mfcc=128, melkwargs={'n_mels':128, 'win_length':frame_length, 
            'hop_length':hop_length, 'n_fft':1024})(wav_data[start:(start+timespan)])
            features[i, 0, :, :] = mfcc.T

            if i % 10000 == 0:
                print('{} pieces have been processed and {} are left.'.format(i, tensor_number-i))

    elif method == 'sole':
        for i in range(tensor_number):
            head = 62*hop_length
            tail = 63*hop_length
            wav_piece = wav_data[0][i*frame_length:(i+1)*frame_length]
            wav_for_val = torch.tensor(np.pad(wav_piece, (head, tail), mode='symmetric'))
            mfcc = transforms.MFCC(n_mfcc=128, melkwargs={'n_mels':128, 'win_length':frame_length, 
            'hop_length':hop_length, 'n_fft':1024})(wav_for_val)
            
            features[i, 0, :, :] = mfcc.T

            if i % 10000 == 0:
                print('{} pieces have been processed and {} are left.'.format(i, tensor_number-i))

    if saving:
        # save_path = r'..\data\for_val\val.npy'
        np.save(save_path, features.numpy())

    print('Feature extraction finished!')

    return features

class CRNN(nn.Module):
    def __init__(self, classnums, hidden_size=1024, num_layers=2):
        super(CRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        cnn = CNN(4)
        cnn.load_state_dict(torch.load(model_cnn))
        cnnlayers = list(cnn.children())
        self.Conv1 = cnnlayers[0]
        self.Conv2 = cnnlayers[1]
        self.Conv3 = cnnlayers[2]
        self.MP = cnnlayers[5]
        self.Dropout1 = cnnlayers[6]
        self.Dropout2 = cnnlayers[7]
        self.Dropout3 = cnnlayers[8]

        cnn_layers = [self.Conv1, self.Conv2, self.Conv3, self.MP, self.Dropout1, self.Dropout2, self.Dropout3]
        for layer in cnn_layers:
            for layer_para in layer.parameters():
                layer_para.requires_grad = False

        self.LSTM = nn.LSTM(input_size=128*32, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.FC1 = nn.Linear(8*self.num_layers*self.hidden_size, 64)
        self.FC2 = nn.Linear(64, classnums)

    def forward(self, x: tensor):
        x = F.relu(self.Conv1(x))
        x = self.MP(self.Dropout1(x))
        x = F.relu(self.Conv2(x))
        x = self.MP(self.Dropout2(x))
        x = F.relu(self.Conv3(x))
        x = self.Dropout3(x)

        x = x.permute([0,2,1,3]).reshape((-1, 8, 128*32))
        x, (h, c) = self.LSTM(x)

        x = x.view(-1, 8*self.num_layers*self.hidden_size)
        x = F.relu(self.FC1(x))
        return self.FC2(x)

if __name__ == '__main__':
    cnn = CNN(4)
    # print(len(list(cnn.children())))
    root_path = '../data'
    model_cnn = '../saved_models/cnn.pkl'
    model_crnn = '../saved_models/crnn.pkl'
    frame_length = 160
    feature_file = '../data/crnn_feature_train.npy'
    label_file='../data/crnn_label_train.npy'

    # Setting the network
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss().to(devices)
    crnn = CRNN(classnums=4).to(devices)
    optimizer = optim.Adam([p for p in crnn.parameters() if p.requires_grad], lr=0.00018964, weight_decay=0.0000019156)
    

    # Training section
    train_dir = root_path + '/train/'
    train_csv = root_path + '/tr_piece.csv'
    files = os.listdir(train_dir)
    running_files = [*map(lambda x: train_dir + x, files)]
    # print(files)
    df = pd.read_csv(train_csv)

    for epoch in range(10):
        for i in range(len(files[:])):
            if os.path.exists(model_crnn):
                crnn.load_state_dict(torch.load(model_crnn))

            sound_file = running_files[i]
            id = files[i].split('.')[0]
            labels = df['label_index'][df['id'] == id].values
            labels = np.pad(labels, (1,1), mode='symmetric')
            print(len(labels))
            np.save(label_file, labels)

            fill_and_extract(sound_file, frame_length, 'series', 'symmetric', True, save_path=feature_file)
            crnn_data = crnn_dataset()
            crnn_loader = DataLoader(dataset=crnn_data, batch_size=128, shuffle=True)

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
                    print('[epoch %d, batch_idx %d: loss %.3f'
                        %(epoch, idx, running_loss/400))
                    running_loss = 0.

            torch.save(cnn.state_dict(), model_crnn)





    