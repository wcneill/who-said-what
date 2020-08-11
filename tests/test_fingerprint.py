from wsw.fingerprint import *
import librosa
import numpy as np
import pytest

@pytest.fixture
def fp():
    filename = librosa.example('nutcracker')
    y, sr = librosa.load(filename)
    return Fingerprint(y, sr)


def test_lpfilter(fp):
    filtered = fp.lpfilter(fp.clip, fp.sr)
    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == fp.clip.shape


def test_downsample(fp):
    pass


def test_fft():
    pass


def test_slide():
    pass


def test_extract():
    pass


def test_create_spectrogram():
    pass


def test_get_prints():
    pass
