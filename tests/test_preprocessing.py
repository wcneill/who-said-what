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
    """
    Test to ensure audio file is written to disk in the expected location.
    Ensure
    :return:
    """
    filename = librosa.ex('nutcracker')
    aud, sr = librosa.load(filename)

    rand_loc = ''.join(random.choices(string.ascii_letters, k=6))
    save_to = os.path.join(ROOT_DIR, rand_loc, 'test.wav')

    write_audio(save_to, aud, sr)
    assert os.path.exists(save_to), "Expected path of saved file does not exist"
    assert os.path.isfile(save_to), "File did not save successfully"

    shutil.rmtree(rand_loc)


def test_resample():
    """
    Tests that a properly structured directory of labelled audio files is successfully
    resampled at the desired rate, saved as a different file type, and written to
    the desired new or old location.
    :return:
    """
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

    # verify new files of type '.wav' save in same/old directory
    resample_all(rand_loc1, rand_loc1, sr)
    assert os.path.isfile(os.path.join(sub_dir1, 'test1.wav')), "File not saved to old directory"
    assert os.path.isfile(os.path.join(sub_dir2, 'test2.wav')), "File not saved to old directory"

    # verify new files of '.wav' are saved in new/different directory
    resample_all(rand_loc1, rand_loc2, sr)
    assert os.path.isfile(os.path.join(rand_loc2, sub_dir1, 'test1.wav')), "File not saved to new directory"
    assert os.path.isfile(os.path.join(rand_loc2, sub_dir2, 'test2.wav')), "File not saved to new directory"

    # Delete all generated test directories and files.
    shutil.rmtree(rand_loc1)
    shutil.rmtree(rand_loc2)
