import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from CNN_dataset import *

print(root_path)
print(torch.__version__)
print(torch.cuda.is_available())

class CNN(nn.Module):
    def __init__(self, classnums):
        super(CNN, self).__init__()
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

# root_path = r'E:\projects\ruixinwei\2021rui\2021_ruixinwei_soundrecognition\data'
# feature_file = r'MFCCs_train.npy'
# label_file = r'labels_train.npy'
# cnn_data = CNN_dataset.MFCCDataset(root_path=root_path, feature_file=feature_file, label_file=label_file)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss().to(device)
cnn = CNN(4).to(device)
optimizer = optim.Adam(cnn.parameters(), lr=0.00018964, weight_decay=0.0000019156)
model_path = r'E:\projects\ruixinwei\2021rui\2021_ruixinwei_soundrecognition\data\cnn.pkl'

# for epoch in range(50):
#     running_loss = 0.
#     for idx, data in enumerate(trainloader, 0):
#         inputs, labels = data[0].to(device), data[1].to(device)
        
#         optimizer.zero_grad()
        
#         outputs = cnn(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
        
#         if idx % 100 == 99:
#             print('[epoch %d, batch_idx %d: loss %.3f'
#                  %(epoch, idx, running_loss/400))
#             running_loss = 0.

# torch.save(cnn.state_dict(), model_path)

cnn.load_state_dict(torch.load(model_path))

correct = 0
total = 0
with torch.no_grad():
    for data in valloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the all val inputs: %d %%' % (
    100 * correct / total))