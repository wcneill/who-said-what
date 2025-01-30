import numpy as np
import librosa
import torch

from fingerprint import Fingerprint
import scipy.signal as sig
import pathlib
from typing import AnyStr

from torchvision import transforms
from torch.utils.data import DataLoader


class ClipAudio:

    def __init__(self, target_length, sample_rate):

        self.target = target_length * sample_rate

    def __call__(self, audio: np.ndarray):

        if len(audio) < self.target:
            audio = np.pad(audio, (0, self.target - len(audio)))
        else:
            audio = audio[:self.target]

        return audio


class Spectrogram:
    def __init__(self, n_fft=2048, hop_length=None, win_length=None):
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft

    def __call__(self, audio_in):
        linear_spectrogram = librosa.stft(audio_in, n_fft=self.n_fft, hop_length=self.hop_length,  win_length=self.win_length)
        return librosa.amplitude_to_db(np.abs(linear_spectrogram), ref=np.max)


class MelSpecFromAudio:

    def __init__(self, sample_rate, n_fft=2048):
        self.n_fft = n_fft
        self.sr = sample_rate

    def __call__(self, audio):
        return librosa.feature.melspectrogram(y=audio, sr=self.sr)


class MelSpecFromSpectrogram:

    def __init__(self, sample_rate):
        self.sr = sample_rate

    def __call__(self, spectrogram):
        return librosa.feature.melspectrogram(S=spectrogram, sr=self.sr)


class LowPassFilter:

    def __init__(self, order, sample_rate, cutoff=None):
        self.cutoff = cutoff
        self.sr = sample_rate
        self.order = order

        if self.cutoff is None:
            self.cutoff = self.sr / 2

    def __call__(self, signal):
        sos = sig.butter(N=self.order, Wn=self.cutoff, fs=self.sr, btype='lowpass', output='sos')
        return sig.sosfilt(sos, signal)


class FingerprintSpec:

    def __init__(self, n_bins, alpha=1):
        self.n_bins = n_bins
        self.alpha = alpha

    def __call__(self, spectrogram):
        return Fingerprint.spec_filter(spectrogram, self.n_bins)


class ToTensorImg:

    def __call__(self, img: np.ndarray):

        img = np.atleast_3d(img)
        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img)


class ToFileTransform:

    def __init__(self, folder_name, file_name_no_ext: AnyStr):

        self.path = pathlib.Path(folder_name) / file_name_no_ext.split(".", maxsplit=2)[0]

    def __call__(self, input):
        if isinstance(input, torch.Tensor):
            torch.save(input, self.path / ".pt")
        elif isinstance(input, np.ndarray):
            np.save(str(self.path / ".np"), input)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    PROJ_DIR = "C:\\Users\\wesle\\source\\repos\\who-said-what"

    rate = 22050
    transform = transforms.Compose(
        [
            ClipAudio(3, rate),
            MelSpecFromAudio(sample_rate=rate, n_fft=2048),
            ToTensorImg()
        ]
    )

    file_location = "C:\\Users\\wesle\\OneDrive\\Documents\\Sound Recordings\\elaine_008.m4a"
    audio, sr = librosa.load(file_location, sr=22050)

    spectrogram = transform(audio).squeeze().numpy()
    librosa.display.specshow(spectrogram, x_axis="time", y_axis="mel", sr=sr)
    plt.show()
    print(spectrogram.size)
