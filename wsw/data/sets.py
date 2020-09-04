# OS and I/O
import os
import sys

# Math and ML libraries
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Audio processing libraries
import librosa as li
import soundfile as sf

# Plotting and Displaying
import matplotlib.pyplot as plt
import seaborn as sns

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

    def __init__(self, csv_file, root_dir, transform=None):
        """
        :param csv_file: (string) Path to the csv file with data annotations
        :param root_dir: (string) Directory containing raw audio data
        :param transform: (callable, optional): Optional transform to be applied on a sample
        """

        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        def __len__(self):
            return len(self.data_frame)

        def __getitem__(self, idx):
            if isinstance(idx, Tensor):
                idx = idx.tolist()

            audio_loc = os.path.join(self.root_dir, self.data_frame.iloc[idx, 2])
            audio, sr = li.load(audio_loc)

            if self.transform:
                audio, sr = self.transform(audio, sr)

            fp = Fingerprint()



