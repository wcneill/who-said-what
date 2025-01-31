from wsw.data.transforms import ClipAudio, MelSpecFromAudio, ToTensorImg, Spectrogram, FingerprintSpec
from torchvision.transforms import Compose
import pathlib
import librosa
import pandas as pd
import torch

from definitions import ROOT_DIR

import matplotlib.pyplot as plt


PROJ_DIR = pathlib.Path("/")
SPEC_TYPE = "melspec"
RATE = 22050

spectrogram_map = {
    "spec":        Spectrogram(),
    "melspec":     MelSpecFromAudio(sample_rate=RATE, n_fft=2048),
    "fingerprint": FingerprintSpec(n_bins=6, alpha=1)
}

label_map = {
    "elaine": 0,
    "wes": 1
}

tfm = Compose(
    [
        ClipAudio(target_length=3, sample_rate=RATE),
        spectrogram_map[SPEC_TYPE],
        ToTensorImg()
    ]
)

data_folder = ROOT_DIR / "training_data"
files = list(data_folder.glob('*.m4a'))
csv = pd.DataFrame(index=range(len(files)), columns=["location", "label"])

for i, file in enumerate(files):

    # load and transform audio
    audio, _ = librosa.load(file, sr=RATE)
    spectrogram = tfm(audio)

    # plt.imshow(spectrogram.numpy().transpose((1, 2, 0)))  # C, H, W -> H, W, C
    # plt.show()

    # save spectrogram tensor
    file_name = pathlib.Path(f"{SPEC_TYPE}_{file.stem}.pt")
    save_path = data_folder / file_name
    torch.save(spectrogram, save_path)

    # update annotations
    person = str(file.stem).split("_")[0]
    csv.loc[i] = [file_name, label_map[person]]

csv.to_csv(data_folder / "annotations.csv", header=False, index=False)









