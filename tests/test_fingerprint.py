from wsw.fingerprint import *
import librosa
import numpy as np
import pytest


@pytest.fixture
def fp():
    filename = librosa.example('nutcracker')
    return Fingerprint(filename)


def test_constructor(fp):
    assert fp.signal is not None
    assert fp.sr is not None


def test_log_bin():
    a = np.arange(512)
    n_bins = 6
    assert len(log_bin(a, n_bins)) == n_bins


def test_ft_filter(fp):
    spec = librosa.stft(fp.signal)
    filtered_dft = ft_filter(spec.T[0], 6)
    assert (filtered_dft != 0).any()
    assert (filtered_dft == 0).any()
    if (filtered_dft == 0).all():
        assert (spec.T[0] == 0).all()


def spec_filter(fp):
    spec = librosa.stft(fp.signal)
    filtered = spec_filter(spec, 6)
    assert (filtered != 0).any()
    assert (filtered == 0).any()
