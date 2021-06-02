import numpy as np
import wave


def sound_load(sound_path, start, end):
    """
    Load sound data from the files with specific segment notated by start and end in time.

    Args:
        sound_path: the wav file in need of loading.
        start: the beginning time of the sound segment.
        end: the ending time of the sound segment.

    Returns:
        : a numpy array of sound data
    """
    with wave.open(sound_path, "rb") as f:
        f = wave.open(sound_path)
        n_channels, sampwidth, framerate, frames, _, _ = f.getparams() # get the params of the file

        start, end = start*framerate, end*framerate
        f.setpos(start) # set the beginning pointer
        data = f.readframes(end-start)
    
    data = np.fromstring(data, dtype=np.short)
    return data