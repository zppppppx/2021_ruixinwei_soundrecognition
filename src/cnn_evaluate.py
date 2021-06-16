import torch
import torchaudio
from torchaudio import transforms
from model import CNN
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
    cnn = CNN.CNN(4).to(device)
    model_path = '../saved_models/cnn.pkl'
    cnn.load_state_dict(torch.load(model_path))

    batch_size = 64
    epoch = int(features.shape[0]/batch_size)
    left = int(features.shape[0] - epoch*batch_size)

    predicts = torch.tensor([]).long().to(device)

    for i in range(epoch):
        feature = features[i*batch_size:(i+1)*batch_size].to(device)
        output = cnn(feature)
        _, predicted = torch.max(output.data, 1)
        predicts = torch.cat((predicts, predicted))
    
    if left != 0 :
        feature = features[(i+1)*batch_size:(i+1)*batch_size+left].to(device)
        output = cnn(feature)
        _, predicted = torch.max(output.data, 1)
        predicts = torch.cat((predicts, predicted))

    return predicts


def frame_fuse(predicts, frame_length=128):
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
            end = i
            timestamps.append((start*timespan, end*timespan))
            start = i
            classes.append(cls_now)

    if predicts[-2] == predicts[-1]:
        classes.append(int(predicts[-1]))
        timestamps.append((start*timespan, (i+1)*timespan))
    else:
        classes.append(int(predicts[-1]))
        timestamps.append((i*timespan, np.round((i+1)*timespan, 3)))

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
        fp = open(csv_file, 'w', newline="")
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

    dir_path = '../data/val/'
    # wav_path = '../data/val/_7oWZq_s_Sk.wav'
    Wav_Path = os.listdir(dir_path)

    val_feature_file = '../data/for_val/val.npy'
    saving = True
    frame_length = 320

    for wav_path in Wav_Path:
        # extraction
        features = fill_and_extract(wav_path, frame_length=frame_length, method='sole', padding_mode='symmetric', saving=saving)
        print(wav_path, features.shape)

        # evaluation
        if saving:
            features = torch.from_numpy(np.load(val_feature_file))
        print(wav_path, features.shape)
        predicts = feature_evaluate(features)
        print(wav_path, predicts.shape)

        # fuse the labels and dump into a csv file
        timestamps, classes = frame_fuse(predicts, frame_length=frame_length)

        if wav_path == Wav_Path[0]:
            csv_generate(wav_path, timestamps, classes, True)
        else:
            csv_generate(wav_path, timestamps, classes, False)


    # visualize
    data_processor = data_preprocess.data_preprocess()
    validate_file = 'val_labels.csv'
    output_file = 'for_val/output.csv'
    wav = '_7oWZq_s_Sk.wav'
    start = 0
    end = 60
    fig1 = plt.figure(figsize=(30, 10))
    data_processor.data_visualize(wav, validate_file, start, end)
    fig2 = plt.figure(figsize=(30, 10))
    data_processor.data_visualize(wav, output_file, start, end)