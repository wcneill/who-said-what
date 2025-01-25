import numpy as np
from librosa import feature

import scipy.signal as sig


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

    def __init__(self, n_fft=2048, sample_rate=22050):
        self.n_fft = n_fft
        self.sr = sample_rate

    def __call__(self, audio):
        return feature.melspectrogram(y=audio, sr=self.sr)


class MelSpecFromSpectrogram:

    def __init__(self, sample_rate=22050):
        self.sr = sample_rate

    def __call__(self, spectrogram):
        return feature.melspectrogram(S=spectrogram, sr=self.sr)


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


