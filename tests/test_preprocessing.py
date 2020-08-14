from wsw.preprocessing import *
from definitions import ROOT_DIR

# for generating random test files
import string
import random

# for audio file i/o
import librosa
import os
import shutil


def test_write_audio():
    filename = librosa.ex('nutcracker')
    aud, sr = librosa.load(filename)

    rand_loc = ''.join(random.choices(string.ascii_letters, k=6))
    save_to = os.path.join(ROOT_DIR, rand_loc, 'test.wav')

    write_audio(save_to, aud, sr)
    assert os.path.exists(save_to)

    shutil.rmtree(rand_loc)


def test_resample():
    aud1, sr = librosa.load(librosa.ex('trumpet'))
    aud2, _ = librosa.load(librosa.ex('nutcracker'))

    # setup mock mnist style dataset file structure
    rand_loc1 = ''.join(random.choices(string.ascii_letters, k=6))
    rand_loc2 = ''.join(random.choices(string.ascii_letters, k=6))
    sub_dir1 = os.path.join(ROOT_DIR, rand_loc1, 'l1')
    sub_dir2 = os.path.join(ROOT_DIR, rand_loc1, 'l2')
    save1 = os.path.join(sub_dir1, 'test1.m4a')
    save2 = os.path.join(sub_dir2, 'test2.m4a')

    # write '.m4a' data to mock file structure
    write_audio(save1, aud1, sr)
    write_audio(save2, aud2, sr)

    # verify new files of type '.wav' save in same directory
    resample_all(rand_loc1, rand_loc1, sr)
    assert os.path.isfile(os.path.join(sub_dir1, 'test1.wav'))
    assert os.path.isfile(os.path.join(sub_dir2, 'test2.wav'))
    # shutil.rmtree(rand_loc1)

    # verify new files of '.wav' are saved in new/different directory
    resample_all(rand_loc1, rand_loc2, sr)
    assert os.path.isfile(os.path.join(rand_loc2, sub_dir1, 'test1.wav'))
    assert os.path.isfile(os.path.join(rand_loc2, sub_dir2, 'test2.wav'))

    # Delete all generated test files.
    shutil.rmtree(rand_loc1)
    shutil.rmtree(rand_loc2)
