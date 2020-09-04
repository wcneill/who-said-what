from wsw.data.fingerprint import Fingerprint
from wsw.data.sets import *
import pytest

import pandas as pd
import librosa as li
import os


@pytest.fixture
def testset():

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

    root = os.path.dirname(f1)

    df.to_csv('test.csv', index=False)
    return AudioImageSet('test.csv', root)


def test_get_item(testset):
    assert len(testset) == 3, \
        'Test set does not contain 3 audio samples (it should)'
    assert testset[0]['image'].shape == (3, testset.size[0], testset.size[1]), \
        "Sample image size does not meet AudioImageSet parameter `size`"



