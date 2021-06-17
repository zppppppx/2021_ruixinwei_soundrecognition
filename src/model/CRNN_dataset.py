from numpy.core.shape_base import stack
import torch
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset
import numpy as np
import os



class crnn_dataset(Dataset):
    def __init__(self, root_path='../data/', feature_file='../data/crnn_feature_train.npy', label_file='../data/crnn_label_train.npy'):
        self.root_path = root_path
        self.feature_file = feature_file
        self.label_file = label_file
        self.features = torch.from_numpy(np.load(self.feature_file))
        self.labels = torch.from_numpy(np.load(self.label_file)).long()

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return(len(self.labels))


