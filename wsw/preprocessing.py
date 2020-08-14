# Project root directory
from definitions import ROOT_DIR

# File and I/O
import warnings
import os

# Audio I/O and processing
import librosa
import soundfile as sf

# Multi-processing tools
from multiprocessing import Pool
import matplotlib.pyplot as plt


# Load an audio file in librosa from file path
def read_audio(file_path, sr=22050):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return librosa.load(file_path, sr=sr)


# write audio to file. Input is tuple (write_path, audio)
def write_audio(path, audio, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with sf.SoundFile(path, 'w', sr, channels=1, format='WAV') as f:
        f.write(audio)


# rename extension
def rename(file_path):
    return os.path.splitext(file_path)[0] + '.wav'


def resample_all(old_path, new_path, sr):
    """
    Re-sample audio data, save it in .wav format and move it to a new folder,
    allowing for two distinct data- sets and/or easy deletion of the original data.

    Audio files are expected to be stored in labeled sub-directories, i.e.
    `project/data/cat_sounds/`, `project/data/dog_sounds/`, `project/data/bird_sounds/`.

    You then point this method at the directory containing the labeled folders by way
    of the 'curr_path' argument. It is then re-sampled and converted to .wav format and
    moved (still in labelled folders) to the directory corresponding to `new_path`.

    Both `curr_path` and `new_path` are relative to the project root.

    :param old_path: path, relative to project root. This directory should contain
    a set of sub-directories corresponding to labelled audio data.
    :param new_path: path, relative to project root. This directory will be created
    for the user, and will receive the re-sampled data in labelled sub-directories.
    This allows the user to easily delete the original audio data if they wish.
    :param sr: The desired rate to re-sample at
    :return: None
    """

    folders = [d for d in os.scandir(old_path) if os.path.isdir(d.path)]
    n_folders = len(folders)

    for i, folder in enumerate(folders):
        dirname = folder.name

        print(f'Working on folder {dirname}')
        print(f'Progress: {int(100 * i / n_folders)}%')

        files = [f.name for f in os.scandir(folder)]
        renamed = map(rename, files)
        r_paths = [f.path for f in os.scandir(folder)]
        w_paths = [os.path.join(new_path, dirname, f) for f in renamed]

        with Pool(os.cpu_count()) as p:
            print('Loading and Resampling Files...')
            data = p.starmap(read_audio, zip(r_paths, [sr]*len(r_paths)))
            print('Saving new files...')
            aud, srs = zip(*data)
            p.starmap(write_audio, zip(w_paths, aud, srs))


if __name__ == '__main__':
    from_path = os.path.join(ROOT_DIR, 'recordings')
    to_path = os.path.join(ROOT_DIR, 'resampled')
    resample_all(from_path, to_path, 100)

    # Visually sure resample data looks like original:
    a, _ = librosa.load(os.path.join(ROOT_DIR, 'resampled\\folder1\\high_pitch.wav'))
    b, _ = librosa.load(os.path.join(ROOT_DIR, 'recordings\\folder1\\high_pitch.m4a'))
    plt.plot(a)
    plt.show()
    plt.plot(b)
    plt.show()
