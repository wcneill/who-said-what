import numpy as np
from scipy.signal import find_peaks

import matplotlib.pyplot as plt


def get_waves(amps, freqs, phase=0):
    """
    Generate wave objects of given frequency and amplitude
    """
    waves = []
    for f, a in zip(freqs, amps):
        w = Wave(a, f, phase=phase)
        waves.append(w)

    return waves


class WaveEncoder:
    def __init__(self):
        self.signal = None
        self.time = None
        self.rate = None
        self.period = None
        self.components = None
        self.fs = None
        self.es = None
        self.idxs = None

    def fit(self, sig, s_rate, s_period, s_time, threshold=None, plot=True):
        self.signal = sig
        self.rate = s_rate
        self.period = s_period
        self.time = s_time
        self.components = self.decompose(sig, s_period, threshold=threshold)
        if plot:
            self.plot_fit()
        return self

    def transform(self, signal, positive_only=True):
        """
        Returns frequencies and amplitudes of transformation domain and range
        """
        N = len(signal)
        e = np.fft.fft(signal) / N
        e = np.abs(e)
        f = np.fft.fftfreq(N, self.period)

        if positive_only:
            e = e[range(int(N / 2))]
            f = f[range(int(N / 2))]

        return e, f

    def decompose(self, signal, s_period, threshold=None):
        """
        Decompose and return the individual components of a composite wave
        form. Plot each component wave.
        """
        es, fs = self.transform(signal, s_period)
        self.es, self.fs = es, fs

        self.idxs, _ = find_peaks(es, threshold=threshold)
        amps, freqs = es[self.idxs], fs[self.idxs]

        return get_waves(amps, freqs)

    def plot_signal(self):
        try:
            N = self.rate * self.time
            t_vec = np.arange(N) * self.period
            plt.plot(t_vec, self.signal, '.')
            plt.title('Signal')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.show()
        except ValueError:
            print('plt_original: Encoder must be fit against '
                  'composite wave before plotting')
            raise

    def plot_components(self):
        colors = plt.rcParams['axes.prop_cycle']()
        try:
            N = self.rate * self.time
            t_vec = np.arange(N) * self.period
            fig, axes = plt.subplots(len(self.components), 1)
            if len(self.components) == 1:
                axes = [axes]

            fig.suptitle('Pure Sine Components')
            for i, wave in enumerate(self.components):
                c = next(colors)["color"]
                axes[i].plot(t_vec, wave.sample(self.rate, self.period, self.time), color=c)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.show()
        except TypeError:
            print('plt_components: Encoder must be fit against composite wave before plotting')
            raise

    def plot_transform(self):
        try:
            plt.title('Fourier Transform')
            plt.plot(self.fs, self.es, 'b.--', label='Energy vs Frequency')
            plt.plot(self.fs[self.idxs],
                     self.es[self.idxs],
                     'ro', label=f'Peak Frequencies:\n{self.fs[self.idxs]}')
            plt.xlabel('Frequency')
            plt.ylabel('Frequency Strength')
            plt.gca().set_xscale('log')
            plt.legend(), plt.grid()
            plt.show()
        except ValueError:
            print('plot_energy: Encoder must be fit against composite wave before plotting')
            raise

    def plot_fit(self):
        self.plot_signal()
        self.plot_transform()
        self.plot_components()


class Wave:
    def __init__(self, amplitude, frequency, phase=0):
        self.amp = amplitude
        self.freq = frequency
        self.phase = phase

    def sample(self, rate, period, time):
        N = rate * time
        t_vec = np.arange(N) * period
        return self.amp * np.sin(2 * np.pi * self.freq * t_vec + self.phase)


if __name__ == '__main__':

    signal_fs = [4, 2, 5]
    signal_as = [1, 2, 3]

    # Sample rate, sample period, time to sample
    Fs = 100
    T = 1 / Fs
    t = 4.5

    # 'Sample' signal
    N = Fs * t
    t_vec = np.arange(N) * T
    signal = 0
    for f, a in zip(signal_fs, signal_as):
        signal += a * np.sin(2 * np.pi * f * t_vec)

    encoder = WaveEncoder()
    # encoder.plot_components()
    # encoder.plot_transform()
    encoder.plot_signal()
    encoder.fit(signal, Fs, T, t, threshold=None)
