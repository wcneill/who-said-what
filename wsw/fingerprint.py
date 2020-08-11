import scipy.signal as sig
import librosa
import numpy as np

def denoise(clip):
    pass


def lpfilter(clip, sr):
    pass


def downsample(clip, sr):
    pass


def stft(signal, window):
    pass


def ffilter(spec, bands=6):
    pass


def spec_filter():
    pass

class Fingerprint():
    def __init__(self, clip, sr, window):
        self.clip = clip
        self.sr = sr
        self.window = window
        self.fingerprint = self.get_prints(self.clip, window)

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


