from numpy.core.shape_base import stack
import torch
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class data_resolute(object):
    """
    This class is used for process data for CNN input.
    """
    def __init__(self, root_path, feature_file, label_file):
        self.root_path = root_path
        self.feature_file = os.path.join(self.root_path, feature_file)
        self.label_file = os.path.join(self.root_path, label_file)

    @staticmethod
    def features_and_labels(soundfile, frag_length=128):
        """
        Using torchaudio.transforms to grab MFCC features and labels which fit pytorch

        Args:
            soundfile: input path of a soundfile
            frag_length: reference length which is used to cut the MFCC features
        
        Returns:
            MFCC: tensor, MFCC features with shape (-1, 128, 128) and the first dimension is the number of data pieces.
            labels: tensor, each element marks one fragment's class, the length is the same as MFCC's first dimension.
        """
        label = soundfile.split('\\')[-1].split('_')[0]
        waveform, sample_rate = torchaudio.load(soundfile)
        MFCCs = transforms.MFCC(n_mfcc=128, melkwargs={'n_mels':128, 'win_length':320, 'hop_length':160, 'n_fft':1024 })(waveform[0][:])
        MFCCs = MFCCs.T.view((-1, frag_length, 128)) # transform the shape into (index, time_representation, melbands)

        frag_nums = MFCCs.shape[0]
        labels = int(label)*np.ones(frag_nums, dtype=np.int8)
        labels = torch.from_numpy(labels)

        return MFCCs, labels


    def dir_resolution(self, src_path, frag_length=128):
        """
        Resolute the whole directory. For each file we adopt features_and_labels to grab the MFCCSs and labels, then
        we concatenate the MFCCs and labels, after which we shuffle them in Dataloader for better representation and training 
        effects.

        Args:
            src_path: str, directory consisting of wav files in need.
            frag_length: int, the length with which we view as a frame.

        
        """
        src_path = os.path.join(self.root_path, src_path)
        files = os.listdir(src_path)

        MFCCs = None
        labels = None
        cnt = 1
        total_num = len(files)
        for wav in files:
            wav_path = os.path.join(src_path, wav)
            MFCCs_each, labels_each = self.features_and_labels(wav_path, frag_length)
            if MFCCs is not None:
                MFCCs = torch.cat((MFCCs, MFCCs_each))
                labels = torch.cat((labels, labels_each))
            else:
                MFCCs, labels = MFCCs_each, labels_each

            if cnt % 1000 == 0:
                print('{} data pieces have been loaded in and {} are left'.format(cnt, total_num-cnt))
            cnt += 1

        np.save(self.feature_file, MFCCs.numpy()) 
        np.save(self.label_file, labels.numpy())
        print('Loading into files finished!')


    def load_features_labels(self):
        """
        Load features and labels from npy files, transforming into tensor.

        Args:
        
        Returns:
            MFCCs: tensor, concatnated MFCCs from features_and_labels.
            labels: tensor, concatnated labels from features_and_labels.
        """
        MFCCs = torch.from_numpy(np.load(self.feature_file))
        labels = torch.from_numpy(np.load(self.label_file))
        'Loading from files finished!'
        return MFCCs.view(-1,1,128,128), labels.long()
        
        
                
class MFCCDataset(Dataset):
    """
    Dataset for training
    """
    def __init__(self, root_path, feature_file, label_file, 
                opt=data_resolute, train=True):
        # self.root_path = root_path
        # self.feature_file = feature_file
        # self.label_file = label_file
        self.opt = opt(root_path, feature_file, label_file)
        self.src_path = 'train_padded' if train else 'val_padded'
        self.features, self.labels = self.opt.load_features_labels()

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return(len(self.labels))

root_path = r'E:\projects\ruixinwei\2021rui\2021_ruixinwei_soundrecognition\data'
train_file = r'MFCCs_train.npy'
train_label_file = r'labels_train.npy'
val_file = r'MFCCs_val.npy'
val_label_file = r'labels_val.npy'
traindata = MFCCDataset(root_path=root_path, feature_file=train_file, label_file=train_label_file)
trainloader = DataLoader(dataset=traindata, batch_size=256, shuffle=True)
valdata = MFCCDataset(root_path=root_path, feature_file=val_file, label_file=val_label_file)
valloader = DataLoader(dataset=traindata, batch_size=256, shuffle=True)



if __name__ == '__main__':
    root_path = r'E:\projects\ruixinwei\2021rui\2021_ruixinwei_soundrecognition\data'
    feature_file = r'E:\projects\ruixinwei\2021rui\2021_ruixinwei_soundrecognition\data\MFCCs_val.npy'
    label_file = r'E:\projects\ruixinwei\2021rui\2021_ruixinwei_soundrecognition\data\labels_val.npy'
    src_path = r'val_padded'
    opt = data_resolute(root_path, feature_file, label_file)
    opt.dir_resolution(src_path)
    mfcc = MFCCDataset(root_path, feature_file, label_file)
    MFCCs, labels = mfcc[1]
    print(MFCCs.shape, labels)

    
    # root_path = r'E:\projects\ruixinwei\2021rui\2021_ruixinwei_soundrecognition\data'
    # mfcc = MFCCDataset(root_path)
    # features, labels = mfcc[0]
    # print(features, labels)
    # print(len(mfcc))