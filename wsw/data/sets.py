# OS and I/O
import os
# import sys

# Math and ML libraries
import torch
from torch import Tensor
from torch.utils.data import Dataset
from skimage import transform

# Audio processing libraries
import librosa as li

# DataFrame
import pandas as pd

# Internal modules
from wsw.data.fingerprint import Fingerprint
from wsw.data import preprocessing as pre


class AudioImageSet(Dataset):
    """
    Audio data imaging dataset. Contains various visual representations of raw frequency spectrum
    data such as standard and mel-spectrograms as well as a sparse matrix "fingerprint" of
    the standard STFT/spectrogram.
    """

    def __init__(self, csv_file, root_dir, imsize=(257, 460), transform=None):
        """
        :param csv_file: (string) Path to the csv file with data annotations
        :param root_dir: (string) Directory containing raw audio data
        :param imsize: (two tuple) Because each layer of a fingerprint object contains a
            different type of spectrogram, we cannot expect each spec to be the same size.
            In order to combine all three spectrograms into a 3-channel image, they need to
            be of uniform size. That size is dictated by this argument.
        :param transform: (callable, optional): Optional transform(s) to be applied to
            audio before it is fingerprinted
        """

        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.size = imsize
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if isinstance(idx, Tensor):
            idx = idx.tolist()

        audio_loc = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1])
        audio, sr = li.load(audio_loc)

        if self.transform:
            out = self.transform(audio)

        fp = Fingerprint(audio, sr)
        image = torch.tensor([transform.resize(d, self.size) for d in fp.fingerprint])
        speakers = self.data_frame.iloc[idx, -1]

        return {'image': image, 'speakers': speakers}
