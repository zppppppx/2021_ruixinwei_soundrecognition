from numpy.core.fromnumeric import ndim
import torchaudio
from torchaudio import transforms
import librosa
import scipy.io.wavfile as wavfile

torchaudio.set_audio_backend("soundfile")

wav_file = r'E:\projects\ruixinwei\2021rui\2021_ruixinwei_soundrecognition\data\train\_a9SWtcaNj8.wav'
waveform, sample_rate = torchaudio.load(wav_file)
# print(waveform.shape)
# mel_specgram = transforms.MelSpectrogram(sample_rate)(waveform)
MFCC = transforms.MFCC(n_mfcc=64, melkwargs={'n_mels':64})(waveform[:64])
# print(mel_specgram.shape)
print('MFCC is: ', MFCC.shape)

sr, sig = wavfile.read(wav_file)
print(sig)
wav = waveform.numpy().reshape(-1)
print(wav)
mfccs = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=40)
# print(mfccs.shape)