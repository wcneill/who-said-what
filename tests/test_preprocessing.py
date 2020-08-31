from wsw.preprocessing import *
from definitions import ROOT_DIR

# for generating random test files
import string
import random

# for audio file i/o
import librosa
import os
import shutil
from soundfile import SoundFile


def test_resample():
    """
    Test to ensure audio file is resampled and written to disk successfully.
    :return:
    """
    filename = librosa.ex('nutcracker')

    rand_loc = ''.join(random.choices(string.ascii_letters, k=6))
    save_to = os.path.join(ROOT_DIR, rand_loc, 'test.wav')

    resample(filename, save_to, sr=22050)
    assert os.path.exists(save_to), "Expected path of saved file does not exist"
    assert os.path.isfile(save_to), "File did not save successfully"
    assert os.path.exists(os.path.join(ROOT_DIR, 'manifest.txt')), "Resampling log not successfully generated"

    shutil.rmtree(os.path.dirname(save_to))
    os.remove(os.path.join(ROOT_DIR, 'manifest.txt'))


def test_resample_all():
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
    rand_loc1 = os.path.join(ROOT_DIR, rand_loc1)
    rand_loc2 = os.path.join(ROOT_DIR, rand_loc2)
    sub_dir1 = os.path.join(rand_loc1, 'l1')
    sub_dir2 = os.path.join(rand_loc1, 'l2')
    save1 = os.path.join(sub_dir1, 'test1.m4a')
    save2 = os.path.join(sub_dir2, 'test2.m4a')

    # write '.m4a' data to mock file structure
    os.makedirs(sub_dir1, exist_ok=True)
    os.makedirs(sub_dir2, exist_ok=True)

    with SoundFile(save1, 'w', sr, channels=1, format='WAV') as f1:
        f1.write(aud1)
    with SoundFile(save2, 'w', sr, channels=1, format='WAV') as f2:
        f2.write(aud2)

    # verify new files of type '.wav' save in same/old directory
    resample_all(rand_loc1, rand_loc1, sr)
    assert os.path.isfile(os.path.join(sub_dir1, 'rs_test1.wav')), "File not saved to old directory"
    assert os.path.isfile(os.path.join(sub_dir2, 'rs_test2.wav')), "File not saved to old directory"

    # verify new files of '.wav' are saved in new/different directory
    resample_all(rand_loc1, rand_loc2, sr)
    assert os.path.isfile(os.path.join(rand_loc2, sub_dir1, 'rs_test1.wav')), "File not saved to new directory"
    assert os.path.isfile(os.path.join(rand_loc2, sub_dir2, 'rs_test2.wav')), "File not saved to new directory"
    assert os.path.isfile(os.path.join(ROOT_DIR, 'manifest.txt')), "Manifest of resampled files not generated"

    # Delete all generated test directories and files.
    shutil.rmtree(rand_loc1)
    shutil.rmtree(rand_loc2)
    os.remove(os.path.join(ROOT_DIR, 'manifest.txt'))


def test_clip_audio():
    """
    Test to ensure audio file is clipped to correct length and written to disk successfully.
    :return:
    """
    filename = librosa.ex('nutcracker')
    audio, sr = librosa.load(filename)
    length = len(audio) / sr

    clip = length // 2
    extend = int(length * 2)

    aud1, sr1 = clip_audio(audio, clip, sr)
    aud2, sr2 = clip_audio(audio, extend, sr)
    assert len(aud1) / sr1 == clip, "Number of sample points does not meet meet expected length for clipped audio"
    assert len(aud2) / sr2 == extend, "Number of sample points does not meet meet expected length for extended audio"

    rand_loc = ''.join(random.choices(string.ascii_letters, k=6))
    save_to = os.path.join(ROOT_DIR, rand_loc, 'test.wav')

    clip_audio(audio, extend, sr, save_to=save_to)
    assert os.path.exists(save_to), "Save to path was unsuccessful"
    assert os.path.isfile(save_to), "File did not save successfully"

    shutil.rmtree(os.path.dirname(save_to))


def test_clip_all():
    """
    Tests that a properly structured directory of labelled audio files is successfully
    clipped or extended to a uniform length according to method inputs.
    :return:
    """
    aud1, sr = librosa.load(librosa.ex('trumpet'))
    aud2, _ = librosa.load(librosa.ex('nutcracker'))

    # setup mock mnist style dataset file structure
    rand_loc1 = ''.join(random.choices(string.ascii_letters, k=6))
    rand_loc2 = ''.join(random.choices(string.ascii_letters, k=6))
    rand_loc3 = ''.join(random.choices(string.ascii_letters, k=6))
    rand_loc1 = os.path.join(ROOT_DIR, rand_loc1)
    rand_loc2 = os.path.join(ROOT_DIR, rand_loc2)
    rand_loc3 = os.path.join(ROOT_DIR, rand_loc3)
    sub_dir1 = os.path.join(rand_loc1, 'l1')
    sub_dir2 = os.path.join(rand_loc1, 'l2')
    save1 = os.path.join(sub_dir1, 'test1.wav')
    save2 = os.path.join(sub_dir2, 'test2.wav')

    os.makedirs(sub_dir1, exist_ok=True)
    os.makedirs(sub_dir2, exist_ok=True)
    os.makedirs(rand_loc3, exist_ok=True)

    # write audio data to mock file structure
    with SoundFile(save1, 'w', sr, channels=1, format='WAV') as f1:
        f1.write(aud1)
    with SoundFile(save2, 'w', sr, channels=1, format='WAV') as f2:
        f2.write(aud2)

    length = len(aud2) / sr
    clip_to = length // 2

    # verify clipped audio is saved in same/old directory
    clip_all(rand_loc1, rand_loc1, clip_to, sr)
    assert os.path.isfile(os.path.join(sub_dir1, 'test1.wav')), "File not saved to old directory"
    assert os.path.isfile(os.path.join(sub_dir2, 'test2.wav')), "File not saved to old directory"

    # verify new files are saved in new/different directory
    clip_all(rand_loc1, rand_loc2, clip_to, sr)
    assert os.path.isfile(os.path.join(rand_loc2, sub_dir1, 'test1.wav')), "File not saved to new directory"
    assert os.path.isfile(os.path.join(rand_loc2, sub_dir2, 'test2.wav')), "File not saved to new directory"

    # verify that log/manifest of previously clipped files is written to disk
    clip_all(rand_loc1, rand_loc2, clip_to, sr, log=os.path.join(ROOT_DIR, 'manifest.txt'))
    assert os.path.isfile(os.path.join(ROOT_DIR, 'manifest.txt')), "Manifest of resampled files not generated"

    # Verify that previously clipped files are skipped:
    clip_all(rand_loc1, rand_loc3, clip_to, sr, restart=True, log=os.path.join(ROOT_DIR, 'manifest.txt'))
    assert not os.listdir(rand_loc3), "Files erroneously re-written despite being in the manifest/log"

    # Delete all generated test directories and files.
    shutil.rmtree(rand_loc1)
    shutil.rmtree(rand_loc2)
    shutil.rmtree(rand_loc3)
    os.remove(os.path.join(ROOT_DIR, 'manifest.txt'))
