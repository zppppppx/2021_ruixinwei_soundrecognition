import numpy as np
from scipy.io import wavfile
import librosa


def sound_load(sound_path, start, end):
    """
    Load sound data from the files with specific segment notated by start and end in time.
    And then extract the features in mfccs.

    Args:
        sound_path: the wav file in need of loading.
        start: the beginning time of the sound segment with unit sec.
        end: the ending time of the sound segment with unit sec.

    Returns:
        mfccsscaled: a numpy array of feature
    """
    try:
        sample_rate, sig = wavfile.read(sound_path)
        start, end = int(start*sample_rate), int(end*sample_rate)
        segment = sig[start:end] # get the fraction of the sound
        if segment.dtype == np.int16:
            bitwidth = 16
        segment = segment/(2.0**bitwidth)
        mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=24)
        mfccsscaled = np.mean(mfccs.T,axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", sound_path)
        return None
    
    return mfccsscaled
