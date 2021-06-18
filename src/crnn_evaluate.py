import torch
from torch import tensor
import torchaudio
from torchaudio import transforms
from model import CRNN
import numpy as np
import csv
from utils import data_preprocess
import matplotlib.pyplot as plt
import os

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
            print(wav_piece.shape)
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


def feature_evaluate(features):
    """
    Input features extracted frame by frame into our model and predict its class, then we write the result into
    a csv file.

    Args:
        features: tensor, returned by fill_and_extract(), size is [frame_size, channel=1, 128, 128].
        frame_length: int, the length of a frame, the same as the arg in fill_and_extract().

    Returns:
        predicts: tensor, labels for every frame.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    crnn = CRNN.CRNN2(2).to(device)
    model_path = '../saved_models/crnn_Conv128_40_mel40.pkl'
    crnn.load_state_dict(torch.load(model_path))

    batch_size = 64
    epoch = int(features.shape[0]/batch_size)
    left = int(features.shape[0] - epoch*batch_size)

    predicts = torch.tensor([]).long().to(device)

    for i in range(epoch):
        feature = features[i*batch_size:(i+1)*batch_size].to(device)
        output = crnn(feature)
        _, predicted = torch.max(output.data, 1)
        predicts = torch.cat((predicts, predicted))
    
    if left != 0 :
        feature = features[(i+1)*batch_size:(i+1)*batch_size+left].to(device)
        output = crnn(feature)
        _, predicted = torch.max(output.data, 1)
        predicts = torch.cat((predicts, predicted))

    return predicts


def smooth(predicts: 'tensor', box: int=3):
    """
    Smooth the results to generate a more continuous result.

    Args:
        predicts: tensor, predicted results by the network.
        box: length of the box, 3 default.
    
    Returns:
        smoothed_predicts: the results after being smoothed, the same length of input predicts.
    """
    padding = int(box/2)
    for_smooth = np.pad(predicts.numpy(), (padding, padding), 'symmetric')
    for i in range(len(predicts)):
        piece = for_smooth[i:i+box]
        predicts[i] = 1 if sum(piece) >= int(box/2)+1 else 0

    return predicts

def frame_fuse(predicts, frame_length=160):
    """
    Since the predicts marks every frame, but we need to get a csv file like train_labels.csv, so we
    need to fuse the time segments into a whole set. Through this function we will get two lists, one's
    each elemnet marking each segment's start and end, the other marking its class.

    Args:
        predicts: tensor or numpy array, the predicted results for every frame.
        frame_length: int, denotes the length of every frame.

    Returns:
        timestamps: a list of tuples, every element is (start, end).
        classes: a list or numpy array, every element is a class matching every element in timestamps.
    """
    timespan = np.round(frame_length/16000, 3)
    timestamps = []
    classes = []
    start = 0
    end = 0

    for i in range(len(predicts[:-1])):
        cls_now = int(predicts[i])
        cls_after = int(predicts[i+1])
        if cls_now != cls_after:
            end = i+1
            timestamps.append((start*timespan, end*timespan))
            start = i+1
            classes.append(cls_now)

    if predicts[-2] == predicts[-1]:
        classes.append(int(predicts[-1]))
        timestamps.append((start*timespan, (i+2)*timespan))
    else:
        classes.append(int(predicts[-1]))
        timestamps.append(((i+1)*timespan, np.round((i+2)*timespan, 3)))

    return timestamps, classes


def csv_generate(sound_file: str, timestamps: list, classes: list, mode: bool, csv_file='../data/for_val/val_pre.csv'):
    """
    Write the results into csvfiles.

    Args:
        sound_file: str, the same as the fill_and_extract.
        timestamps: a list of tuples, every element is (start, end).
        classes: a list or numpy array, every element is a class matching every element in timestamps.
        csv_file: destination csv file we want to write into
    """
    Main_classes = ['CLEAN_SPEECH', 'SPEECH_WITH_MUSIC', 'SPEECH_WITH_NOISE', 'NO_SPEECH']
    if mode:
        fp = open(csv_file, 'a', newline="")
        fp_writer = csv.writer(fp)
        fp_writer.writerow(['id', 's', 'e', 'label', 'label_index'])
        id = sound_file.split('/')[-1].split('.')[0]
        for i in range(len(classes)):
            s, e = str(timestamps[i][0]), str(timestamps[i][1])
            label = Main_classes[classes[i]]
            label_index = str(classes[i])
            fp_writer.writerow([id, s, e, label, label_index])
        fp.close()
        print('CSV file has been generated successfully!')

    else:
        fp = open(csv_file, 'a', newline="")
        fp_writer = csv.writer(fp)
        id = sound_file.split('/')[-1].split('.')[0]
        for i in range(len(classes)):
            s, e = str(timestamps[i][0]), str(timestamps[i][1])
            label = Main_classes[classes[i]]
            label_index = str(classes[i])
            fp_writer.writerow([id, s, e, label, label_index])
        fp.close()
        print('CSV file has been generated successfully!')


if __name__ == '__main__':

    dir_path = 'E:/projects/ruixinwei/2021rui/test/'
    # wav_path = '../data/val/_7oWZq_s_Sk.wav'
    # wav_path = '../data/val/o4xQ-BEa3Ss.wav'
    Wav_Path = os.listdir(dir_path)

    val_feature_file = 'E:/projects/ruixinwei/2021rui/test.npy'
    saving = True
    frame_length = 128
    csv_file = 'E:/projects/ruixinwei/2021rui/val_pre.csv'

    if os.path.exists(csv_file):
        os.remove(csv_file)

    for i in range(len(Wav_Path)):
        wav_path = dir_path + Wav_Path[i]
        # extraction
        features = fill_and_extract(wav_path, frame_length=frame_length, method='series', padding_mode='symmetric', saving=saving, save_path=val_feature_file)
        print(wav_path, features.shape)

        # evaluation
        if saving:
            features = torch.from_numpy(np.load(val_feature_file))
        print(wav_path, features.shape)
        predicts = feature_evaluate(features).cpu()
        predicts = smooth(predicts, 5)
        predicts[predicts == 0] = 3

        print(wav_path, predicts.shape)

        # fuse the labels and dump into a csv file
        timestamps, classes = frame_fuse(predicts)

        mode = True if i == 0 else False
        csv_generate(wav_path, timestamps, classes, mode, csv_file)




    # # visualize
    # data_processor = data_preprocess.data_preprocess()
    # validate_file = 'val_labels.csv'
    # output_file = 'for_val/val_pre.csv'
    # wav = '_7oWZq_s_Sk.wav'
    # # wav = 'o4xQ-BEa3Ss.wav'
    # start = 0
    # end = 120

    # data_processor.data_visualize(wav, validate_file, output_file,  start, end)