import scipy.signal as sig
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


class Fingerprint:
    """
    This is a simple class that contains class variables describing audio data. These
    class variables can be thought of as 'fingerprints' of the original audio signal.
    Currently, the fingerprint class stores two forms of the data: Original signal and
    a filtered spectrogram which is a sparse matrix describing the most dominant frequencies
    of the original signal.

    :param audio_path: The path from which to load audio data for fingerprinting
    :param sr: The desired sample rate to load audio data at
    :param n_fft: The number of DFTs to use in creating the STFT/spectrogram
        fingerprint of the original audio data.
    """
    def __init__(self, audio_path, n_fft=512):
        y, sr = librosa.load(audio_path, sr=None)
        self.sr = None
        self.signal = y
        self.n_fft = n_fft
        self.fingerprint = self.get_prints(self.signal, sr, n_fft)

    def show(self):
        D = librosa.amplitude_to_db(self.fingerprint)
        img = librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=self.sr)
        plt.show()

    def get_prints(self, signal, sr, n_fft):
        signal = Fingerprint._lpfilter(signal, sr)
        signal = Fingerprint._downsample(signal, sr, new_sr=11025)
        spec = Fingerprint.stft(signal, n_fft)
        spec = Fingerprint.denoise(spec)
        spec = Fingerprint.spec_filter(spec, 6)
        self.sr = 11025
        return spec

    @staticmethod
    def stft(signal, N):
        """
        Compute the STFT/spectrogram of given signal with Hamming Window

        :param signal: The audio signal to perform STFT on.
        "param N: The number of ffts
        """
        spec = librosa.stft(signal, n_fft=N, window=sig.windows.hamming)
        return np.abs(spec)

    @staticmethod
    def ft_filter(ft, n_bins, alpha=1):
        """
        Filters out weaker frequencies from a fourier transform. Could be thought of as analogous to
        using Scipy.signal.find_peaks, then converting all non peak data to zeros.

        Peak data is determined by first dividing the frequency bins into logarithmic bands. Then, the
        strongest frequency from each band is identified. Finally, these strongest frequencies are
        averaged. All frequencies that fall below this average times a constant, alpha, are converted
        to zeros.

        :param ft: The fourier transform to filter.
        :param n_bins: The number of logarithmic bands to divide the frequency domain into
        :param alpha: Coefficient for determining the threshold at which data is dropped.
        """
        binned_ft = Fingerprint._log_bin(ft, n_bins)
        for i, band in enumerate(binned_ft):
            band_max = np.max(band)
            filtered = np.where(band < band_max, 0, band)
            binned_ft[i] = filtered
        joined_filtered = np.concatenate(binned_ft)
        thresh = np.mean(joined_filtered)

        return np.where(joined_filtered < thresh * alpha, 0, joined_filtered)

    @staticmethod
    def spec_filter(spec, n_bins):
        """
        Iteratively filter the DFTs that make up a spectrogram, leaving a sparse matrix
        representing the strongest frequencies over the time domain. See `Fingerprint.ft_filter`
        for how this is achieved.

        :param spec: The spectrogram to filter.
        :param n_bins: The number logarithmic bands to divide the frequency domain into.
        """
        filtered = np.zeros_like(spec.T)
        for i, dft in enumerate(spec.T):
            filtered[i] = Fingerprint.ft_filter(dft, n_bins)
        return filtered.T

    @staticmethod
    def _lpfilter(signal, sr):
        """Helper method. Applies Low-pass filter that attenuates at 10kHz"""
        cutoff = 10e3
        sos = sig.butter(10, cutoff, fs=sr, btype='lowpass', analog=False, output='sos')
        return sig.sosfilt(sos, signal)

    @staticmethod
    def _log_bin(arr, n_bins):
        """Helper method. Divide spectrogram frequency bins logarithmically"""
        bands = np.array([10 * 2 ** i for i in range(n_bins - 1)])
        idxs = np.arange(len(arr))
        split_arr = np.split(arr, np.searchsorted(idxs, bands))
        return split_arr

    @staticmethod
    def denoise(spectrogram):
        """Noise filter"""
        return librosa.decompose.nn_filter(spectrogram)

    @staticmethod
    def _downsample(signal, old_sr, new_sr):
        """Helper Method. resample data."""
        return librosa.resample(signal, old_sr, new_sr)


if __name__ == '__main__':
    path = librosa.ex('trumpet')
    fp = Fingerprint(path)
    fp.show()
