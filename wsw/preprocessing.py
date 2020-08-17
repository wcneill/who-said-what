# Project root directory
from definitions import ROOT_DIR

# File and I/O
import warnings
import os
import sys

# Audio I/O and processing
import librosa
import soundfile as sf

# Multi-processing tools
from multiprocessing import Pool


def resample(old_path, new_path, sr, ext='WAV', manifest=None):
    """
    Re-samples an audio file and writes it to a new location as a .wav file. This method also keeps a
    log of previously re-sampled files. This is useful for the case when a large number of files are
    being re-sampled, and the process is paused or interrupted. Using the generated 'manifest.txt', a
    user would be able to pick up where the re-sampling process left off.

    :param old_path: Path where original file exists.
    :param new_path: Path to write to. Will be created if it doesn't exist. Will be overwritten if it does.
    :param sr: The sampling rate of the audio
    :param ext: the output format of the saved file. Default `WAV`. Use
        `soundfile.available_formats()`
    :return:
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        audio, sr = librosa.load(old_path, sr=sr)

    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    with sf.SoundFile(new_path, 'w', sr, channels=1, format=ext) as f:
        f.write(audio)

    # write filename to manifest as completed.
    if manifest is not None:
        fname = os.path.basename(old_path)
        with open(manifest, 'a') as manifest:
            manifest.write(fname + '\n')


def rename(file_path, ext='.wav'):
    """
    Renames a file by prepending it with "rs_" to signify that it has been resampled, and
    change the extension to match the `ext` argument (default .wav).

    :param file_path:
    :param ext:
    :return:
    """
    return 'rs_' + os.path.splitext(file_path)[0] + ext


def resample_all(old_loc, new_loc, sr, restart=False, manifest=None):
    """
    Re-sample audio data, save it in .wav format and move it to a new folder,
    allowing for two distinct data- sets and/or easy deletion of the original data.

    Audio files are expected to be stored in labeled sub-directories, i.e.
    `project/data/cat_sounds/`, `project/data/dog_sounds/`, `project/data/bird_sounds/`.

    You then point this method at the directory containing the labeled folders by way
    of the 'curr_path' argument. It is then re-sampled and converted to .wav format and
    moved (still in labelled folders) to the directory corresponding to `new_path`.

    Both `curr_path` and `new_path` are relative to the project root.

    :param old_loc: path, relative to project root. This directory should contain
        a set of sub-directories corresponding to labelled audio data.
    :param new_loc: path, relative to project root. This directory will be created
        for the user, and will receive the re-sampled data in labelled sub-directories.
        This allows the user to easily delete the original audio data if they wish.
    :param sr: The desired rate to re-sample at
    :param restart: If true, manifest keyword argument is mandatory. If true
        this method will search through a log of previously resampled files
        in order to prevent them from being resampled again. This is handy
        if the er-sample process was interrupted part way through.
    :param manifest: An optional argument giving a file path to a log for recording
        what files have been successfully resampled and saved.
    :return: None
    """

    folders = [d for d in os.scandir(old_loc) if os.path.isdir(d.path)]

    for i, folder in enumerate(folders):
        sys.stdout.flush()
        sys.stdout.write(f'Working on {i} of {len(folders)}')
        sys.stdout.write(f'{int(100 * i / len(folders))}% complete. ')
        dirname = folder.name

        if restart:
            with open(manifest) as m:
                completed = [line.strip()[:-4] for line in m]
            files = [f for f in os.scandir(folder) if (f.name[:-4] not in completed) and os.path.isfile(f.path)]
        else:
            files = [f for f in os.scandir(folder) if os.path.isfile(f.path)]

        fpaths = [f.path for f in files]
        fnames = [f.name for f in files]
        renamed = map(rename, fnames)

        new_paths = [os.path.join(new_loc, dirname, f) for f in renamed]

        with Pool(os.cpu_count() - 1) as p:
            p.starmap(resample, zip(fpaths, new_paths, [sr] * len(fpaths)))

def create_manifest(fpath, mpath):
    """
    Generates a log of pre-existing files in a data directory. Used to recover interrupted
    re-sampling if a log is not already present.

    :param fpath: The location of the files you would like to add to the log.
    :param mpath: The location you would like to save the log to.
    """
    folders = [f for f in os.scandir(fpath) if os.path.isdir(f.path)]
    for folder in folders:
        files = [fi.name + '\n' for fi in os.scandir(folder) if os.path.isfile(fi.path)]
        with open(mpath, 'a') as manifest:
            manifest.writelines(files)

