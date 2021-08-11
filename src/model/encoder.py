import torch
import os
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchaudio
import torchaudio.transforms as transforms
import matplotlib.pyplot as plt

def awgn(sig, snr, dim=0, method='vectorized'):
    """
    This function realizes adding noise to a signal array with Gaussian noise.

    Args:
        sig: tensor, the audio signal that is needed to be noised.
        snr: float, the ratio of signal's power to noise's power, in unit of dB.
        dim: int, denoting the position of the signal, this is decided by the input shape of sig.

    Returns:
        sig_noised: noised signal.
    """
    dims = len(list(sig.size()))
    length = sig.shape[dim]
    if method == 'vectorized':
        N = sig.numel()
        Ps = torch.sum(sig**2/N)

    elif method == 'max_en':
        Ps = torch.max(torch.sum(sig**2/length, dim=dim))

    elif method == 'axial':
        Ps = torch.sum(sig**2/length, dim=dim)

    else:
        raise ValueError('method \"' + str(method) + '\" not recognized.')

    # The power of signal in db
    Psdb = 10 * torch.log10(Ps)
    if dims == 1 or dims == 2:
        Psdb = Psdb.reshape([1])

    # The power of noise in db
    Pndb = Psdb - snr

    std_norml = torch.randn(sig.shape)
    noise = torch.unsqueeze(torch.sqrt(10 ** (Pndb/10)), dim=dim)*std_norml

    return noise + sig



class enc_data(Dataset):
    def __init__(self, audio_path, snr=50, sr=48000, root_path='../data/') -> None:
        """
        Initialize the class.
        """
        self.root_path = root_path
        self.audio_path = os.path.join(root_path, audio_path)
        frags = self.audio_load()
        self.data, self.marks = self.mel_gen(frags, snr, sr)


    def audio_load(self):
        """
        This function realizes the function of loading the audio pieces into a tensor.

        Args:

        Returns:
            frags: the audio pieces in the shape of [batchsize, channel, length].
        """
        files = os.listdir(self.audio_path)
        file_num = 0
        total_num = len(files)
        
        frags = torch.zeros([total_num, 1, 58560])
        for each_file in files:
            file_path = os.path.join(self.audio_path, each_file)
            frag, sr = torchaudio.load(file_path)
            frags[file_num,:] = frag
            # frag = frag[None, :]
            # frags = torch.cat((frags, frag), dim=0)
            file_num += 1

            if file_num % 1000 == 999:
                print('%d pieces have been loaded in and %d are left.' % (file_num+1, total_num-file_num))

        print('All files have been loaded in!')
        return frags


    def mel_gen(self, frags, snr, sr):
        """
        This function realizes the function of extracting the mel spectrum without and with noise.

        Args:
            frags: input audio frags with the shape of [batchsize, channel, length]

        Returns:
            mel_origin: the mel spectrum of original audio signals.
            mel_noised: the mel spectrum of noised audio signals.
        """
        frags_noised = awgn(frags, snr, dim=2, method='axial')
        mel_spec = transforms.MelSpectrogram(sample_rate=sr, n_fft=400*3, win_length=320*3, hop_length=244*3, n_mels=64)

        mel_noised = mel_spec(frags_noised)+0.000001
        mel_std = mel_spec(frags)+0.000001 # This is for avoiding NaN loss
        return mel_noised.log2(), mel_std.log2()


    def __getitem__(self, index):
        return self.data[index], self.marks[index]

    
    def __len__(self):
        return self.data.shape[0]



class enc_dec(nn.Module):
    def __init__(self):
        super(enc_dec, self).__init__()

        self.enc = nn.Sequential(
            # Feature extraction layer
            nn.Conv2d(1, 32, 3, padding=1), # batch*32*64*81
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), # batch*64*64*81
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # Dimension reduction layer
            nn.MaxPool2d((4,3)), # batch*64*16*27
            nn.Conv2d(64, 128, 3, padding=1), # batch*128*16*27
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2,3)), # batch*128*8*9
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128,64,3,(2,3),1,(1,2)), # batch*64*16*27
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,64,3,1,1), # batch*64*16*27
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,32,3,(4,3),1,(3,2)), # batch*32*64*81
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32,1,3,1,1) # batch*1*64*81
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x





    

    
        
if __name__ == '__main__':
    # setting basic parameters
    enc_path = 'piece/'
    model_path = '../saved_models/enc_dec.pkl'
    batch_size = 200
    lr = 1e-5
    epoch = 100
    
    # enc_dataset = enc_data(enc_path)
    # enc_dataloader = DataLoader(dataset=enc_dataset, batch_size=batch_size, shuffle=True)

    # frags = enc_dataset.audio_load()
    # print(frags.shape)

    # mel_n, mel = enc_dataset.mel_gen(frags, 50, 48000)
    # print(mel_n.shape)



    # # set up the network
    # devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # criterion = nn.MSELoss().to(devices)
    # enc_dec_net = enc_dec().to(devices)
    # if os.path.exists(model_path):
    #     enc_dec_net.load_state_dict(torch.load(model_path))
    # optimizer = optim.Adam(enc_dec_net.parameters(), lr=lr)


    # # Training
    # for i in range(epoch):
    #     running_loss = 0.
    #     for idx, data in enumerate(enc_dataloader, 0):
    #         mel_noised, mel_std = data[0].to(devices), data[1].to(devices)
    #         optimizer.zero_grad()

    #         outputs = enc_dec_net(mel_noised)
    #         loss = criterion(outputs, mel_std)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()

    #         if idx % 20 == 19:
    #             print('epoch: %d, bactch index: %d, loss: %.4f' % (i, idx, running_loss/20))
    #             running_loss = 0.

    #     if i % 10 == 9:
    #         torch.save(enc_dec_net.state_dict(), model_path)

    # torch.save(enc_dec_net.state_dict(), model_path)
    # enc_dec_net = None

    # Visualize the effects
    filedir = '../data/train/'
    # filedir = '../data/enc/'
    filename = '2DUITARAsWQ.wav'
    # filename = 'ARA NORM  0002.wav'
    wav_file = filedir + filename
    enc_dec_net = enc_dec()
    enc_dec_net.load_state_dict(torch.load(model_path))

    audio, sr = torchaudio.load(wav_file)
    noised_audio = awgn(audio, snr=50, dim=1)

    mel_spec = transforms.MelSpectrogram(sample_rate=sr, n_fft=400, win_length=320, hop_length=244, n_mels=64)
    start = int(16000*180)
    mel_audio = (mel_spec(audio[:,start:start+320*61])+0.000001).log2()
    mel_noised = (mel_spec(noised_audio[:,start:start+320*61])+0.000001).log2()

    net_input = mel_audio[None, :]
    with torch.no_grad():
        net_output = enc_dec_net(net_input)
    net_output = torch.squeeze(net_output, dim=0)
    net_output.requires_grad = False

    plt.figure
    plt.subplot(3, 1, 1)
    plt.imshow(mel_audio[0])
    plt.title('Original')
    plt.subplot(3, 1, 2)
    plt.imshow(mel_noised[0])
    plt.title('Noised')
    plt.subplot(3, 1, 3)
    plt.imshow(net_output[0])
    plt.title('Denoise')
    plt.show()
