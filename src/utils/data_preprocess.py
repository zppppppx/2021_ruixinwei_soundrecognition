from librosa import feature
import numpy as np
from numpy.lib.function_base import extract
from scipy.fftpack.realtransforms import dst
import torchaudio
from torchaudio import transforms
import librosa
import pandas as pd
import os
import csv

class data_preprocess:
    """
    Define the class of data preprocess to better resolute the data.
    Including feature_extract, csv_resolution
    """
    def  __init__(self, root_path='../data'):
        """
        Initialize the args

        Args:
            root_path: the path of the data source
        """
        self.root_path = root_path


    def sound_frag_load(self, filename, start=0, end=-1):
        """
        Simply load a piece of data with timestamps input.

        Args:
            filename: the wav file in need of reading.
            start: the start of the sound fragment, in unit of sec.
            end: the end of the sound fragment, in unit of sec.

        Return:
            segment: a numpy array consisting of a sound fragment.
            sample_rate: a number marking the sample rate.
        """
        sig, sample_rate = torchaudio.load(filename)
        sig = sig.numpy().reshape(-1)

        if end == -1:
            return sample_rate, sig
        
        start, end = int(start*sample_rate), int(end*sample_rate)
        segment = sig[start:end]
        return sample_rate, segment


    def feature_extract(self, sample_rate, sig, start, end):
        """
        Load sound data from the files with specific segment notated by start and end in time.
        And then extract the features of mfccs.

        Args:
            sound_path: the wav file in need of loading.
            start: the beginning time of the sound segment with unit sec.
            end: the ending time of the sound segment with unit sec.

        Returns:
            mfccsscaled: a numpy array of feature
        """
        # get the fraction of the sound
        start, end = int(start*sample_rate), int(end*sample_rate)
        segment = sig[start:end] 
        
        # unify the sound to [0, 1]
        sampwidth = [np.int8, np.int16, np.int32]
        bitwidth = sampwidth.index(segment.dtype) + 1
        segment = segment/(2.0**bitwidth)

        # extract the mfcc feature
        mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
        return mfccsscaled


    def csvfile_resolution(self, csv_path):
        """
        Analyze the csv file aiming to get wav filename 
        and the corresponding audio segment split up with time mark.

        Args:
            csv_path: the path of the csv file, only the filename needed.

        Return:
            filenames: a list of names of files. eg: [file1, ...]
            timestamps: a list of timestamp lists marking the start and the end of a segment. eg: [[(s1,e1), (s2,e2)], ...]
            labels: a list of label lists marking the class corresponding to timestamp. eg: [[class1, class2], ...]
        """
        csv_path = os.path.join(self.root_path, csv_path)
        raw_data = pd.read_csv(csv_path)
        filenames = list(set(raw_data['id']))

        timestamps = []
        labels = []
        file_nums = len(raw_data)

        cnt = 0
        for filename in filenames:
            datapiece = raw_data[['s', 'e', 'label_index']][raw_data['id'] == filename]
            timestamp = []
            label = []

            # get the corresponding timestamp and labels
            for i in range(len(datapiece)):
                stamp = datapiece[['s', 'e']].iloc[i].values
                timestamp.append(np.round(stamp, 2))
                label.append(datapiece['label_index'].iloc[i])
                
                cnt += 1
                if cnt % 20000 == 0:
                    print("{} pieces have been loaded in and {} are left".format(cnt, file_nums-cnt))

            timestamps.append(np.array(timestamp, dtype=np.float32))
            labels.append(np.array(label, dtype=np.int8))
        print('Resolution finished!')
        
        return filenames, timestamps, labels

    
    def feature_to_file(self, filenames, timestamps, labels, dst_path='mfcc_train.txt', train=True):
        """
        Extract the feature of the sound fragment and save it to a txt or csv file.

        Args:
            train: marking your target dir, True or false
            filenames: a list of filenames in need of extraction.
            timestamps: similar to csvfile_resolution.
            labels: similar to csvfile_resolution.
            dst_path: the dst file you want to dump the results.
        """
        # handle the path issues
        dir = 'train' if train else 'val'
        dst_path = os.path.join(self.root_path, dst_path)
        if os.path.exists(dst_path):
            os.remove(dst_path)

        with open(dst_path, "a") as f:
            # walk through all the files
            cnt = 0
            for i in range(len(filenames)):
                filename = filenames[i]
                sound_path = os.path.join(self.root_path, dir, filename+'.wav')
                try:
                    sig, sample_rate = torchaudio.load(sound_path)
                    sig = sig.numpy().reshape(-1)
                
                except Exception as e:
                    print("Error encountered while parsing file: ", sound_path)
                    return None
                
                timestamp = timestamps[i]
                label = labels[i]

                # walk through all the labels
                for j in range(len(label)):
                    mfcc_features = self.feature_extract(sample_rate, sig, timestamp[j][0], timestamp[j][1])
                    classification = label[j]
                    row_data = str(classification) + '\t'
                    # elements = [str(element) + ' ' for element in mfcc_features]
                    for element in mfcc_features:
                        row_data = row_data + str(element) + ' '
                    row_data += '\n'

                    f.write(row_data)

                    cnt += 1
                    if cnt%100 == 0:
                        print('{} sound fragments processed'.format(cnt))

        print('File has been created')


    def feature_load(self, filename='mfcc_train.txt'):
        """
        Load data from feature file.

        Args:
            filename: the file saving features.

        Return:
            labels: numpy arrays marking the labels
            features: numpy arrays of features
        """
        filename = os.path.join(self.root_path, filename)

        labels = []
        features = []
        cnt = 0
        with open(filename, 'r') as f:
            data = f.readlines()
            total_num = len(data)
            for row_data in data:
                row_data = row_data.rstrip()
                label, feature = row_data.split('\t')
                feature = feature.split(' ')
                feature = [float(i) for i in feature]
                feature = np.array(feature)

                features.append(feature)
                labels.append(label)

                cnt += 1
                if cnt % int(0.001*total_num) == 0:
                    print('{0:.1f}%% features have been loaded in and {1:.1f}%% are left.'.\
                        format(cnt/total_num*100, (total_num-cnt)/total_num*100))

        return  np.array(labels), np.array(features)

    def frame_resolution(self, filenames, timestamps, labels, sr=16000, span=160, dst_path='tr_piece.csv'):
        """
        Identify a group of frames' label and save it to a csv file

        Args:
            filenames: similar to csvfile_resolution.
            timestamps: similar to csvfile_resolution.
            labels: similar to csvfile_resolution.
            sr: sample rate of the wav file, 16000 default.
            span: the number of frames that you want to categorize, 160 default which is 0.01 sec.
            dst_path: the dst file you want to dump the results.
        """
        classes = ['CLEAN_SPEECH', 'SPEECH_WITH_MUSIC', 'SPEECH_WITH_NOISE', 'NO_SPEECH']
        dst_path = os.path.join(self.root_path, dst_path)

        with open(dst_path, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['id', 's', 'e', 'label', 'label_index'])
        
            # walk through all the files
            cnt = 0
            for i in range(len(filenames)):
                filename = filenames[i]
                timestamp = timestamps[i]
                label = labels[i]

                # walk through all the labels
                for j in range(len(label)):
                    lb = label[j]
                    s, e = timestamp[j]
                    number_frags = round((e-s)*sr/span)

                    # walk through each sound segment and separate it into frames pieces
                    for k in range(number_frags):
                        s_k = s + k*(span/sr)
                        e_k = s_k + span/sr
                        csv_writer.writerow([filename, str(s_k), str(e_k), classes[lb], str(lb)])
                        cnt += 1

                        if cnt%20000 == 0:
                            print('{} sets of frames have been processed'.format(cnt))

        print('Segments resoluted! And the total number of  fragments is {}'.format(cnt))
                    