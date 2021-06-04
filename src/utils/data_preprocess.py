import numpy as np
from numpy.lib.function_base import extract
from scipy.io import wavfile
import librosa
import pandas as pd
import os

class data_preprocess:
    """
    Define the class of data preprocess to better resolute the data.
    Including sound_load, csv_resolution
    """
    def  __init__(self, root_path):
        """
        Initialize the args

        Args:
            root_path: the path of the data source
        """
        self.root_path = root_path


    def sound_load(self, sound_path, start, end):
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
        try:
            # get the fraction of the sound
            sample_rate, sig = wavfile.read(sound_path)
            start, end = int(start*sample_rate), int(end*sample_rate)
            segment = sig[start:end] 
            
            # unify the sound to [0, 1]
            sampwidth = [np.int8, np.int16, np.int32]
            bitwidth = sampwidth.index(segment.dtype) + 1
            segment = segment/(2.0**bitwidth)

            # extract the mfcc feature
            mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=24)
            mfccsscaled = np.mean(mfccs.T,axis=0)

        except Exception as e:
            print("Error encountered while parsing file: ", sound_path)
            return None
        
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



    