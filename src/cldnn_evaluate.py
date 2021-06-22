import torch
import torchaudio

import numpy as np
import csv
from utils import data_preprocess
from model.CLDNN import CLDNN
import matplotlib.pyplot as plt
import os

def fill_cut(wav_file: str):
    """
    In order to evaluate the file, we need to pad the audio file to a suitable length, which 
    is compatible of audio_length/frame_length

    Args:
        wav_file: the filename we want to process.

    Returns:
        frags: tensor, processed frags, every element contains an audio segment and its
        neighbors, which we set both 10 frames.
    """
    waveform, sr = torchaudio.load(wav_file)
    waveform = waveform[0].numpy()
    padded_waveform = np.pad(waveform, (10*320, 10*320 + 160), mode='symmetric')

    frags = np.zeros((90000, 1, 320*21))
    for i in range(90000):
        frags[i, 0] = padded_waveform[i*160:i*160+320*21]
    
    return torch.tensor(frags).float()

def evaluate(frags, model:str):
    """
    Evaluate the input frags and return the predicted classes.

    Args:
        frags: tensor, used for input into the network.
        model: the model we need to evaluate the inputs.

    Returns
        predicts: tensor, predicted results.
    """
    batch_size = 512
    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    cldnn = CLDNN().to(device)
    cldnn.load_state_dict(torch.load(model))

    epoch = int(frags.shape[0]/batch_size)
    left = int(frags.shape[0] - epoch*batch_size)

    predicts = torch.tensor([]).long().to(device)
    with torch.no_grad():
        for i in range(epoch):
            frag = frags[i*batch_size:(i+1)*batch_size].to(device)
            output = cldnn(frag)
            _, predicted = torch.max(output.data, 1)
            predicts = torch.cat((predicts, predicted))
        
        if left != 0 :
            frag = frags[(i+1)*batch_size:(i+1)*batch_size+left].to(device)
            output = cldnn(frag)
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
    for_smooth = np.pad(predicts.cpu().numpy(), (padding, padding), 'symmetric')
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
            timestamps.append((np.round(start*timespan, 3), np.round(end*timespan,3 )))
            start = i+1
            classes.append(cls_now)

    if predicts[-2] == predicts[-1]:
        classes.append(int(predicts[-1]))
        timestamps.append((np.round(start*timespan, 3), np.round((i+2)*timespan, 3)))
    else:
        classes.append(int(predicts[-1]))
        timestamps.append((np.round((i+1)*timespan, 3), np.round((i+2)*timespan, 3)))

    return np.array(timestamps), np.array(classes)

def csv_generate(sound_file: str, timestamps: list, classes: list, mode: bool, csv_file='../data/for_val/val_pre.csv'):
    """
    Write the results into csvfiles.

    Args:
        sound_file: str, the same as the fill_and_extract.
        timestamps: a list of tuples, every element is (start, end).
        classes: a list or numpy array, every element is a class matching every element in timestamps.
        csv_file: destination csv file we want to write into
    """

    # Main_classes = ['CLEAN_SPEECH', 'SPEECH_WITH_MUSIC', 'SPEECH_WITH_NOISE', 'NO_SPEECH']
    if mode:
        fp = open(csv_file, 'a', newline="")
        fp_writer = csv.writer(fp)
        fp_writer.writerow(['id', 's', 'e'])
        id = sound_file.split('/')[-1].split('.')[0]
        for i in range(len(classes)):
            s, e = str(timestamps[i][0]), str(timestamps[i][1])
            # label = Main_classes[classes[i]]
            # label_index = str(classes[i])
            fp_writer.writerow([id, s, e])
        fp.close()
        print('CSV file has been generated successfully!')

    else:
        fp = open(csv_file, 'a', newline="")
        fp_writer = csv.writer(fp)
        id = sound_file.split('/')[-1].split('.')[0]
        for i in range(len(classes)):
            s, e = str(timestamps[i][0]), str(timestamps[i][1])
            # label = Main_classes[classes[i]]
            # label_index = str(classes[i])
            fp_writer.writerow([id, s, e])
        fp.close()
        print('CSV file has been generated successfully!')



if __name__ == '__main__':
    # wav_file = r'E:\projects\ruixinwei\2021rui\2021_ruixinwei_soundrecognition\data\val\J1jDc2rTJlg.wav'
    dir_path = '../data/val/'
    csv_file = '../data/for_val/val_pre_test.csv'
    dir_path = 'E:/projects/ruixinwei/2021rui/test/'
    csv_file = 'E:/projects/ruixinwei/2021rui/test_pre.csv'
    Wav_Path = os.listdir(dir_path)

    # set up the network
    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    model_cldnn = '../saved_models/final.pkl'
    



    if os.path.exists(csv_file):
        os.remove(csv_file)

    for i in range(len(Wav_Path)):
        wav_file = dir_path + Wav_Path[i]
        # Evaluation
        frags = fill_cut(wav_file)
        # print(frags.shape)
        predicts = evaluate(frags, model_cldnn)
        # predicts[predicts<3] = 1
        # predicts[predicts==3] = 0
        predicts = smooth(predicts, 5)
        predicts[predicts == 0] = 3

        timestamps, classes = frame_fuse(predicts)
        # print(classes, timestamps)
        timestamps = timestamps[classes != 3]
        classes = classes[classes != 3]
        # print(classes, timestamps)
        mode = True if i == 0 else False
        csv_generate(wav_file, timestamps, classes, mode, csv_file)

        print('The {}th audio file has been evaluated! And {} are left'.format(i+1, len(Wav_Path)-i-1))
    
    # # visualize
    # data_processor = data_preprocess.data_preprocess()
    # validate_file = 'val_labels.csv'
    # wav = '_7oWZq_s_Sk.wav'
    # # wav = 'o4xQ-BEa3Ss.wav'
    # start = 0
    # end = 120

    # data_processor.data_visualize(wav, validate_file, csv_file,  start, end)
