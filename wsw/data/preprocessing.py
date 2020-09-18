# Project root directory
from definitions import ROOT_DIR

# File and I/O
import warnings
import os

# Audio I/O and processing
import librosa
import soundfile as sf

# math
import numpy as np

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
    :param manifest: Path to .txt or other file where you wish to write completed work.
    :return:
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        audio, sr = librosa.load(old_path, sr=sr)

    if not os.path.exists(os.path.dirname(new_path)):
        try:
            os.makedirs(os.path.dirname(new_path))
        except FileExistsError:
            pass
    with sf.SoundFile(new_path, 'w', sr, channels=1, format=ext) as f:
        f.write(audio)

    # write filename to manifest as completed.
    fname = os.path.basename(old_path)
    if manifest is not None:
        with open(manifest, 'a') as manifest:
            manifest.write(fname + '\n')
    else:
        with open(os.path.join(ROOT_DIR, 'manifest.txt'), 'a') as m:
            m.write(fname + '\n')


def rename(file_path, ext='.wav'):
    """
    Renames a file by prepending it with "rs_" to signify that it has been resampled, and
    change the extension to match the `ext` argument (default .wav).

    :param file_path: file path to rename with .wav (or other) extension
    :param ext: The replacement extension
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

    if restart:
        with open(manifest) as m:
            completed = set([line.strip() for line in m])

    folders = [d for d in os.scandir(old_loc) if os.path.isdir(d.path)]

    for i, folder in enumerate(folders):
        print(f'Working on folder {folder.name}. Overall progress: {int(100 * i / len(folders))}%   \r', end="")
        dirname = folder.name

        if restart:
            files = [f for f in os.scandir(folder) if (f.name not in completed) and os.path.isfile(f.path)]
        else:
            files = [f for f in os.scandir(folder) if os.path.isfile(f.path)]

        fpaths = [f.path for f in files]
        fnames = [f.name for f in files]
        renamed = map(rename, fnames)

        new_paths = [os.path.join(new_loc, dirname, f) for f in renamed]

        with Pool(os.cpu_count() - 1) as p:
            n = len(fpaths)
            p.starmap(resample, zip(fpaths, new_paths, [sr] * n, ['WAV'] * n, [manifest] * n))


def interval_pad(audio, interval, sr=22050):
    """
    Extend an audio to a length such that len(audio) mod some time interval is
    equal to zero. This method is intended to be useful in the case where
    one wishes to create a sequence of equal length chunks of an audio
    signal.

    :param audio: The audio to pad. Audio will be converted to numpy
        array.
    :param interval: The time interval in seconds. The returned audio
        will be such that len(audio) / interval = 0.
    :param sr: The sampling rate of the audio file. Default is 22050 to
        match the Librosa library defaults.
    :return: NumPy array of padded audio data. Returns audio un-modified
        if the length of the audio is equal to the interval input.
    """

    audio = np.array(audio)
    a_samples = len(audio)
    i_samples = sr * interval

    if (a_samples % i_samples) == 0:
        return audio
    if a_samples < i_samples:
        diff = i_samples - a_samples
        return np.concatenate((audio, np.zeros(diff)), axis=None)
    if a_samples > i_samples:
        rem = a_samples % i_samples
        diff = i_samples - rem
        return np.concatenate((audio, np.zeros(diff)), axis=None)


def sequence(audio, interval, sr=22050):
    """
    Create a sequence of equal sized audio samples from a single audio clip.
    Will raise an error if the interval chosen does not split the audio evenly.
    In this case, consider using `interval_pad` to prepare the audio for sequencing.

    :param audio: The audio to split. Should be a Numpy array
    :param interval: The time interval in seconds to divide the audio clip by
    :param sr: Sample rate of audio. Default is 22050 to match Librosa defaults.
    """
    n_sections = int(len(audio) / (interval * sr))
    return np.split(audio, n_sections)


def clip_audio(audio, length, sr=22050, save_to=None, log=None):
    """
    clip or extend a signal by either cutting it short or padding it with zeros.

    :param audio: a 1D numpy array representing the signal you wish to clip or pad.
    :param length: Length to clip or pad audio to. Audio longer than this value
        will be clipped and audio shorter than this value will be padded.
    :param sr: The sample rate of audio signal
    :param save_to: Location to save clipped audio to.
    :param log: Optional log file location to track files that have already
        been saved to file
    """

    m_samples = len(audio)
    n_keep = int(length * sr)

    if m_samples > n_keep:
        audio = audio[:n_keep]
    if m_samples < n_keep:
        to_add = n_keep - m_samples
        audio = np.concatenate((audio, np.zeros(to_add)))

    if save_to is not None:
        if not os.path.exists(os.path.dirname(save_to)):
            print(os.path.exists(os.path.dirname(save_to)))
            print('Why are you here if the path exists???')
            try:
                os.makedirs(os.path.dirname(save_to))
            except FileExistsError:
                pass
            with sf.SoundFile(save_to, 'w', sr, channels=1) as f:
                f.write(audio)

        if log is not None:
            with open(log, 'a') as m:
                m.write(os.path.basename(save_to) + '\n')
    else:
        return audio, sr


def clip_all(fpath, save_to, length, sr=None, restart=False, log=None):
    """
    Clip or pad all files audio files found in the umbrella directory `fpath`
    to a single desired length, then save to a new (or same) location. Files
    will be saved in the `.wav` format.

    :param fpath: The directory containing audio files in labelled folders. In
    other words, the directory `fpath` should contain sub-directories where
    each sub-directory is a unique label for the data within.
    :param save_to: The location to save the clipped files too. Original
        labelled file structure will be preserved.
    :param length: The length to cut or pad variable length audio signal to.
    :param sr: The sample rate of the audio being fingerprinted.
    :param restart: If true then the `log` argument is mandatory. When true,
        this method will look for a log file containing a list of already
        fingerprinted files.
    :param log: The path to the log which tracks the already completed files.
        This argument is mandatory if `restart=True`.
    """

    if restart:
        with open(log) as lg:
            completed = set([line.strip() for line in lg])

    folders = [d for d in os.scandir(fpath) if os.path.isdir(d.path) if not d.name.startswith('.')]
    print(folders)

    for i, folder in enumerate(folders):
        dirname = folder.name
        # print(f'Working on folder {dirname}. Overall progress: {int(100 * i / len(folders))}%   \r', end="")

        if restart:
            files = [f for f in os.scandir(folder) if (f.name not in completed) and os.path.isfile(f.path)]
        else:
            files = [f for f in os.scandir(folder) if os.path.isfile(f.path)]

        fpaths = [f.path for f in files]
        fnames = [f.name for f in files]

        if files:
            new_paths = [os.path.join(save_to, dirname, f) for f in fnames]

            with Pool(os.cpu_count() - 1) as p:
                n = len(files)
                z = p.starmap(librosa.load, zip(fpaths, [sr] * n))
                aud, srs = zip(*z)
                p.starmap(clip_audio, zip(aud, [length] * n, srs, new_paths, [log] * n))

    print('\nResizing of all audio files complete.')


# def fingerprint_all(fpath, save_to, length, sr, restart=False, log=None):
#     """
#     Fingerprint all files audio files found in the umbrella directory `fpath`.
#
#     :param fpath: The directory containing audio files in labelled folders. In
#     other words, the directory `fpath` should contain sub-directories where
#     each sub-directory is a unique label for the data within.
#     :param length: The length to cut or pad variable length audio signal to.
#     :param sr: The sample rate of the audio being fingerprinted.
#     :param restart: If true then the `log` argument is mandatory. When true,
#         this method will look for a log file containing a list of already
#         fingerprinted files.
#     :param log: The path to the log which tracks the already completed files.
#         This argument is mandatory if `restart=True`.
#     """
#     if restart:
#         with open(log) as lg:
#             completed = set([line.strip() for line in lg])
#
#         folders = [d for d in os.scandir(fpath) if os.path.isdir(d.path)]
#
#     for i, folder in enumerate(folders):
#         dirname = folder.name
#         print(f'Working on folder {dirname}. Overall progress: {int(100 * i / len(folders))}%   \r', end="")
#
#
#         if restart:
#             files = [f for f in os.scandir(folder) if (f.name not in completed) and os.path.isfile(f.path)]
#         else:
#             files = [f for f in os.scandir(folder) if os.path.isfile(f.path)]
#
#         fpaths = [f.path for f in files]
#         fnames = [f.name for f in files]
#
#         new_paths = [os.path.join(save_to, dirname, f) for f in fnames]
#
#         with Pool(os.cpu_count() - 1) as p:
#             N = len(fpaths)
#             p.starmap()
#             p.starmap(resample, zip(fpaths, new_paths, [sr] * N, ['WAV'] * N, [manifest] * N))
