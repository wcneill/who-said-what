import scipy.signal as sig
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


class Fingerprint:
    """
    This is a simple class that contains class variables describing  different views
    of the same audio data.

    The fingerprint object stores four total views of audio data: The original signal,
    a STFT spectrogram, a reduced spectrogram containing only the most powerful
    frequencies, and a mel cepstrum.

    :param audio_path: The path from which to load audio data for fingerprinting
    :param n_fft: The number of DFTs to use in creating the STFT/spectrogram
        fingerprint of the original audio data.
    """
    def __init__(self, audio_path, rsr=11025, n_fft=512):
        self.signal, self.sr = librosa.load(audio_path, sr=None)
        self.n_fft = n_fft
        self.fingerprint = self.get_prints(self.signal, self.sr, rsr, n_fft)

    def show(self):
        """
        This method displays the fingerprint data via matplotlib
        """

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        specs = (librosa.amplitude_to_db(self.fingerprint[i]) for i in range(3))
        scales = ('hz', 'mel', 'hz')

        for i, (sp, sc) in enumerate(zip(specs, scales)):
            librosa.display.specshow(sp, x_axis='time', y_axis=sc, sr=self.sr, ax=axes[i])

        plt.tight_layout()
        plt.show()

    def get_prints(self, signal, sr, rsr, n_fft):

        if rsr:
            signal = Fingerprint._lpfilter(signal, sr, rsr)
            signal, self.sr = librosa.resample(signal, sr, rsr), rsr

        spec = Fingerprint.stft(signal, n_fft)
        sparse_spec = Fingerprint.spec_filter(spec, 6)
        mel_spec = librosa.feature.melspectrogram(S=spec, sr=self.sr)

        return spec, mel_spec, sparse_spec

    @staticmethod
    def stft(signal, N):
        """
        Compute the real valued STFT/spectrogram of given signal with Hamming Window

        :param signal: The audio signal to perform STFT on.
        :param N: The number of ffts
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
    def _lpfilter(signal, sr, rsr):
        """
        Helper method. Attenuates signal frequencies in preparation for down sampling
        in order to prevent aliasing during the downsampling process. For this reason,
        the filter attenuates frequencies that are above the desired resampling rate
        divided by 2.

        :param signal: The signal to filter
        :param sr: The sample rate of the original signal
        :param rsr: The rate to which the original signal will be downsampled.
        """
        cutoff = rsr / 2
        sos = sig.butter(10, cutoff, fs=sr, btype='lowpass', analog=False, output='sos')
        return sig.sosfilt(sos, signal)

    @staticmethod
    def _log_bin(arr, n_bins):
        """
        Helper method. Divide spectrogram frequency bins logarithmically

        :param arr: The array to divide.
        :param n_bins: The number of bins to divide the array into.
        """
        bands = np.array([10 * 2 ** i for i in range(n_bins - 1)])
        idxs = np.arange(len(arr))
        split_arr = np.split(arr, np.searchsorted(idxs, bands))
        return split_arr


if __name__ == '__main__':
    path = librosa.ex('trumpet')
    fp = Fingerprint(path)
    fp.show()
