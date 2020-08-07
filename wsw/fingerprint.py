

# Psueodocode the algorithm
class Fingerprint:
    def __init__(self, clip, window):
        self.clip = clip
        self.fingerprint = self.get_prints(self.clip, window)

    # Apply Low Pass Filter
    def lpfilter(self, clip):
        pass

    # Downsample
    def downsample(self, clip):
        pass

    # Compute FT
    def fft(self, window, signal):
        pass

    # Compute FFTs over sliding window
    def slide(self, clip, window):
        pass

    # divide each FT into logarithmic bands and extract
    # Strongest frequencies. Return filtered transforms
    def extract(self, fts, n_bands):
        pass

    # create spectrogram from set of fourier transforms
    def create_spectrogram(self, fts):
        pass

    # putting it all together
    def get_prints(self, clip, window):
        clip = self.lpfilter(clip)
        clip = self.downsample(clip)
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




