import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from CNN import *

class CRNN(nn.Module):
    def __init__(self, classnums):
        super(CRNN, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(2,2), padding_mode='reflect')
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=(2,2), padding_mode='reflect')
        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=(2,2), padding_mode='reflect')
        self.fc1 = nn.Linear(in_features=128*8*32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=classnums)
        self.MP = nn.MaxPool2d(kernel_size=(4,2))
        self.Dropout1 = nn.Dropout(0.2)
        self.Dropout2 = nn.Dropout(0.3)
        self.Dropout3 = nn.Dropout(0.35)
        self.Dropout4 = nn.Dropout(0.5)

if __name__ == '__main__':
    cnn = CNN(4)
    print(list(cnn.children()))