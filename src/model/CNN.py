import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_size, classnums):
        super(CNN, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2, padding_mode='reflect')
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, padding_mode='reflect')
        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, padding_mode='reflect')
        self.fc1 = nn.Linear(in_features=128*8*32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=4)
        self.MP = nn.MaxPool2d(kernel_size=(4,2))
        self.Dropout1 = F.dropout(0.2)
        self.Dropout2 = F.dropout(0.3)
        self.Dropout3 = F.dropout(0.35)
        self.Dropout4 = F.dropout(0.5)

    def forward(self, x):
        x = F.relu(self.Conv1(x))
        x = self.MP(self.Dropout1(x))
        x = F.relu(self.Conv2(x))
        x = self.MP(self.Dropout2(x))
        x = F.relu(self.Conv3(x))
        x = self.Dropout3(x)

        x = x.view(-1, 128*8*32)
        x = F.relu(self.fc1(x))
        x = self.Dropout1(x)
        
        return self.fc2(x)
