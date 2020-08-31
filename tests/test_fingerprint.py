from wsw.data.fingerprint import *
import librosa
import numpy as np
import pytest


@pytest.fixture
def fp():
    filename = librosa.example('nutcracker')
    return Fingerprint(filename)


def test_constructor(fp):
    """Test that instance variables are filled"""
    assert fp.signal is not None, "No signal received/initialized in constructor"
    assert fp.sr is not None, "No sample rate registered in constructor"


# noinspection PyProtectedMember
def test_log_bin():
    """
    Method tests that the correct number of bins is created
    """
    a = np.arange(512)
    n_bins = 6
    assert len(Fingerprint._log_bin(a, n_bins)) == n_bins, "array divided into incorrect number of bins"


def test_ft_filter(fp):
    """
    This test ensures that the returned fourier transform has non-zero entries, or that
    if all entries are zero, then the original transform was also all zero.
    :param fp: fingerprint object
    :return:
    """
    spec = librosa.stft(fp.signal)
    filtered_dft = fp.ft_filter(spec.T[0], 6)
    assert (filtered_dft == 0).any(), "No zero valued elements, but filter is expected to return a sparse vector"
    if (filtered_dft == 0).all():
        assert (spec.T[0] == 0).all(), "Filtered DFT is empty despite non-empty DFT input to the filter"


def test_spec_filter(fp):
    """
    Assures that a filtered spectrogram has a mix of non-zero and zero valued entries,
    as `spec_filter` is expected to return a sparse matrix (fingerprint object).
    :param fp:
    :return:
    """
    spec = librosa.stft(fp.signal)
    filtered = fp.spec_filter(spec, 6)
    assert (filtered == 0).any(), "No zero valued elements but filter is expected to return sparse matrix"
    if (filtered == 0).all():
        assert (spec == 0).all(), "Filtered spectrogram is empty despite non-empty spectrogram input"
