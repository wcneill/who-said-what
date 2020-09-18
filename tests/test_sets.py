from wsw.data.sets import *
import pytest

import pandas as pd
import librosa as li
import os


@pytest.fixture
def rootdir():

    f1 = li.ex('trumpet')
    f2 = li.ex('nutcracker')
    f3 = li.ex('vibeace')
    df = pd.DataFrame({
        'Title': ['a', 'b', 'c'],
        'URL': [None, None, None],
        'Filename': [f1, f2, f3],
        'Date': [None, None, None],
        'Speakers': [1, 2, 3]
    })

    df.to_csv('test.csv', index=False)
    return os.path.dirname(f1)


def test_ais_get_item(rootdir):
    ais = AudioImageSet('test.csv', rootdir)
    assert len(ais) == 3, \
        'Test set does not contain 3 audio samples (it should)'
    assert ais[0].shape == (3, ais.size[0], ais.size[1]), \
        "Sample image size does not meet AudioImageSet parameter `size`"
