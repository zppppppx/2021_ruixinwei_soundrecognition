import torch
import torchaudio

from model import CNN
import numpy as np
import csv
from utils import data_preprocess
import matplotlib.pyplot as plt
import os

def fill_cut(wav_file: str):
    """
    In order to evaluate the file, we need to pad the audio file to a suitable length, which 
    is compatible of audio_length/frame_length

    Args:
        wav_file: the filename we want to
    """
    waveform, sr = torchaudio.load(wav_file)
    waveform = waveform[0].numpy()
    padded_