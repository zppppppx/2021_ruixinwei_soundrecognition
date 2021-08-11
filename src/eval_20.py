import torch
import torchaudio

import numpy as np
import csv
from utils import data_preprocess
# from model.CLDNN_mel import CLDNN_mel
from model.CLDNN_longer_120_withenc import CLDNN_mel
import matplotlib.pyplot as plt
import os
from utils.filter import filter

def fill_cut_eval(wav_file: str, model: str):
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
    padded_waveform = np.pad(waveform, (30*320, 30*320 + 160), mode='symmetric')

    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    cldnn = CLDNN_mel().to(device)
    cldnn.load_state_dict(torch.load(model))

    batch_size = 512
    epoch = int(90000/batch_size)
    left = int(90000 - epoch*batch_size)

    proba = torch.tensor([]).float()
    with torch.no_grad():
        for i in range(epoch):
            frags = torch.zeros(batch_size, 1, 320*61)
            start = i*batch_size
            for j in range(batch_size):
                frags[j,0,:] = torch.tensor(padded_waveform[(start+j)*160:(start+j)*160+320*61])
            frags = frags.to(device)
            output = cldnn(frags)[:, 1].to("cpu")
            frags = None
            proba = torch.cat((proba, output))
            output = None

            # print('Epoch:  %d' % i)
        
        if left != 0:
            frags = torch.zeros(left, 1, 320*61)
            start = epoch*batch_size
            for j in range(left):
                frags[j,0,:] = torch.tensor(padded_waveform[(start+j)*160:(start+j)*160+320*61])
            frags = frags.to(device)
            output = cldnn(frags)[:, 1].to("cpu")
            frags = None
            proba = torch.cat((proba, output))
            output = None

            
        # for i in range(90000):
        #     # print(i)
        #     frags = torch.zeros(1,1,320*31)
        #     frags[0,0,:] = torch.from_numpy(padded_waveform[i*160:i*160+320*31])
        #     frags = frags.float().to(device)
        #     output = cldnn(frags)[:, 1].to("cpu")
        #     frags = None
        #         # _, predicted = torch.max(output.data, 1)
        #     proba = torch.cat((proba, output))
        #     output = None
        #     if i % 10000 == 9999:
        #         print(i)

    proba = filter(proba.to("cpu").numpy(), 101, method='mean')
    proba[proba > 0.5] = 1
    proba[proba <= 0.5] = 0
    
    return proba



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
    # dir_path = 'E:/projects/ruixinwei/2021rui/test/'
    # csv_file = 'E:/projects/ruixinwei/2021rui/test_pre.csv'
    Wav_Path = os.listdir(dir_path)

    # set up the network
    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    model_cldnn = '../saved_models/cldnn_mel_longer_120_with_enc.pkl'
    



    if os.path.exists(csv_file):
        os.remove(csv_file)

    for i in range(len(Wav_Path)):
        wav_file = dir_path + Wav_Path[i]
        # Evaluation
        predicts = fill_cut_eval(wav_file, model_cldnn)

        predicts[predicts == 0] = 3

        timestamps, classes = frame_fuse(predicts)
        # print(classes, timestamps)
        timestamps = timestamps[classes != 3]
        classes = classes[classes != 3]
        # print(classes, timestamps)
        mode = True if i == 0 else False
        csv_generate(wav_file, timestamps, classes, mode, csv_file)
        predicts, timestamps, classes = None, None, None

        print('The {}th audio file has been evaluated! And {} are left'.format(i+1, len(Wav_Path)-i-1))
    
    # visualize
    data_processor = data_preprocess.data_preprocess()
    validate_file = 'val_labels.csv'
    csv_file = 'for_val/val_pre_test.csv'
    wav = '_7oWZq_s_Sk.wav'
    # wav = 'o4xQ-BEa3Ss.wav'
    start = 300
    end = 450

    data_processor.data_visualize(wav, validate_file, csv_file,  start, end)
