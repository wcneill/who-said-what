import numpy as np
import librosa
import torch

from fingerprint import Fingerprint
import scipy.signal as sig
import pathlib
from typing import AnyStr
from torch.utils.data import DataLoader


class ClipAudio:

    def __init__(self, target_length, sample_rate):

        self.target = target_length
        self.sr = sample_rate

    def __call__(self, audio: np.ndarray):

        if len(audio) < self.target:
            audio = np.pad(audio, (0, self.target - len(audio)))
        else:
            audio = audio[:self.target]

        return audio


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


class BinFilterSpec:

    def __init__(self, n_bins, alpha=1):
        self.n_bins = n_bins
        self.alpha = alpha

    def __call__(self, spectrogram):
        return Fingerprint.spec_filter(spectrogram, self.n_bins)


class AsTensor:

    def __call__(self, input: np.ndarray):
        return torch.from_numpy(input)


class ToFileTransform:

    def __init__(self, folder_name, file_name_no_ext: AnyStr):

        self.path = pathlib.Path(folder_name) / file_name_no_ext.split(".", maxsplit=2)[0]

    def __call__(self, input):
        if isinstance(input, torch.Tensor):
            torch.save(input, self.path / ".pt")
        elif isinstance(input, np.ndarray):
            np.save(str(self.path / ".np"), input)


if __name__ == "__main__":
    file_location = "C:\\Users\\wesle\\OneDrive\\Documents\\Sound Recordings\\wes_001.m4a"
    audio, sr = librosa.load(file_location, sr=22050)
    print(sr)