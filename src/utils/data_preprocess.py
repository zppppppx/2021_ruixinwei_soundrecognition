from librosa import feature
import numpy as np
from numpy.lib.function_base import extract
from scipy.fftpack.realtransforms import dst
from scipy.io import wavfile
import librosa
import pandas as pd
import os

class data_preprocess:
    """
    Define the class of data preprocess to better resolute the data.
    Including sound_load, csv_resolution
    """
    def  __init__(self, root_path='../data'):
        """
        Initialize the args

        Args:
            root_path: the path of the data source
        """
        self.root_path = root_path


    def sound_load(self, sample_rate, sig, start, end):
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
        mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=24)
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
        for filename in filenames:
            datapiece = raw_data[['s', 'e', 'label_index']][raw_data['id'] == filename]
            timestamp = []
            label = []

            # get the corresponding timestamp and labels
            for i in range(len(datapiece)):
                stamp = datapiece[['s', 'e']].iloc[i].values
                timestamp.append(np.round(stamp, 2))
                label.append(datapiece['label_index'].iloc[i])

            timestamps.append(np.array(timestamp, dtype=np.float32))
            labels.append(np.array(label, dtype=np.int8))
        
        return filenames, timestamps, labels

    
    def feature_extract(self, filenames, timestamps, labels, dst_path, train=True):
        """
        Extract the feature of the sound fragment and save it to a txt or csv file.

        Args:
            train: marking your target dir, True or false
            filenames: a list of filenames in need of extraction.
            timestamps: similar to csvfile_resolution.
            labels: similar to csvfile_resolution.
            dst_path: the file path to be saved as.
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
                    sample_rate, sig = wavfile.read(sound_path)
                
                except Exception as e:
                    print("Error encountered while parsing file: ", sound_path)
                    return None
                
                timestamp = timestamps[i]
                label = labels[i]

                # walk through all the labels
                for j in range(len(label)):
                    mfcc_features = self.sound_load(sample_rate, sig, timestamp[j][0], timestamp[j][1])
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

    