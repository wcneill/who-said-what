import scipy.signal as sig
import librosa
import numpy as np

def denoise(clip):
    pass


def lpfilter(signal, sr):
    cutoff = 10e3
    sos = sig.butter(10, cutoff, fs=sr, btype='lowpass', analog=False, output='sos')
    return sig.sosfilt(sos, signal)


def downsample(signal, old_sr, new_sr):
    return librosa.resample(signal, old_sr, new_sr)


def stft(signal, N):
    spec = librosa.stft(signal, n_fft=N, window=sig.windows.hamming)
    return np.abs(spec)


def denoise(spectrogram):
    return librosa.decompose.nn_filter(spectrogram)


def log_bin(arr, n_bins):
    bands = np.array([10 * 2 ** i for i in range(n_bins - 1)])
    idxs = np.arange(len(arr))
    split_arr = np.split(arr, np.searchsorted(idxs, bands))
    return split_arr


def ft_filter(ft, n_bins, alpha=1):
    binned_ft = log_bin(ft, n_bins)
    for i, band in enumerate(binned_ft):
        band_max = np.max(band)
        filtered = np.where(band < band_max, 0, band)
        binned_ft[i] = filtered
    joined_filtered = np.concatenate(binned_ft)
    thresh = np.mean(joined_filtered)

    return np.where(joined_filtered < thresh * alpha, 0, joined_filtered)


def spec_filter(spec, n_bins):
    filtered = np.zeros_like(spec.T)
    for i, dft in enumerate(spec.T):
        filtered[i] = ft_filter(dft, n_bins)
    return filtered.T

class Fingerprint():
    def __init__(self, path, sr=22050, n_ftt=512):
        y, sr = librosa.load(path, sr)
        self.signal = y
        self.sr = sr
        self.n_ftt = 512
        # self.fingerprint = self.get_prints(self.clip, window)

    # divide each FT into logarithmic bands and extract
    # Strongest frequencies. Return filtered transforms


    # create spectrogram from set of fourier transforms
    def create_spectrogram(self, fts):
        pass

    # putting it all together
    def get_prints(self, clip, window):
        clip = denoise(clip)
        clip = lpfilter(clip)
        clip = downsample(clip)
        fts = self.slide(clip, window)
        fts = self.extract(fts, n_bands=6)
        return self.create_spectrogram(fts)

# ------------Pseudo-code---------------:

# get soundclip
# apply low pass filter
# downsample
# create sliding window:
    # fts = []
    # for each window:

        # calculate FT
        # add FT to list
    # Return list of FTs.

# For Each FFT:
    # Divide into logarithmic bands
    # Keep strongest freqs from each band
    # average strongest freqs
    # discard freqs with amplitude < average * alpha
    # Get Spectrogram from filtered FFTs

# Use Spectrogram fingerprints as data for ML model.

# ------------- Extra Credit --------------------

# If I want to do the "full" fingerpint look-up table:

# --------- 'Server Side' -----------#
# for each known/labeled audio clip:
    # Get filtered spectrogram from audio
    # Sort points on spectrogram by time then frequency
    # Choose zone anchor point convention
    # Create target zones of spectrogram:
    # For each target zone:
        # For each point in Zone:
            # Create Address for point: [anchor freq, point freq, delta t]
            # Create Value for point: [anchor_time, songID]

# -----------'Client Side' --------------
# Get audio clip
# Apply fingerprint algorithm (get filtered spectrogram)
# Create Target Zones
# for each zone:
    # for each point in zone
        # Create Address [anchor_freq, point_freq, delta_t]
        # Create Value [anchor_time]

# NOTES:
# Librosa has a stft function to create spectrograms.
# Libros also has a 'split' function which splits audio into non-silent
# portions of audio. This would be helpful for the birdcall recordings


