import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from CNN import *

class CRNN(nn.Module):
    def __init__(self, classnums):
        super(CRNN, self).__init__()

if __name__ == '__main__':
    cnn = CNN(4)
    print(list(cnn.children()))