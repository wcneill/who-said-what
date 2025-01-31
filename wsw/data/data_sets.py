# OS and I/O
import pathlib
# import sys

# Math and ML libraries
import torch
from torch import Tensor
from torch.utils.data import Dataset

# Audio processing libraries
import librosa as li

# DataFrame
import pandas as pd

# Internal modules
from wsw.data.fingerprint import Fingerprint


class AudioImageSet(Dataset):
    """
    Audio data imaging dataset. Contains various visual representations of raw frequency spectrum
    data such as standard and mel-spectrograms as well as a sparse matrix "fingerprint" of
    the standard STFT/spectrogram.
    """

    def __init__(self, root_dir, annotations_file, transform=None):
        """
        :param csv_file: (string) Path to the csv file with data annotations
        :param root_dir: (string) Directory containing spectrograms
        :param transform: (callable, optional): Optional transform(s) to be applied to
            audio before it is returned
        """

        self.root_dir = pathlib.Path(root_dir)
        self.data_frame = pd.read_csv(self.root_dir.joinpath(annotations_file))
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if isinstance(idx, Tensor):
            idx = idx.tolist()

        file_name = pathlib.Path(self.data_frame.iloc[idx, 0])

        image = torch.load(self.root_dir / file_name)
        label = self.data_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label}
