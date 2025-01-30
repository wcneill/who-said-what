from wsw.data.transforms import ClipAudio, MelSpecFromAudio, ToTensorImg, Spectrogram, FingerprintSpec
from torchvision.transforms import Compose
import pathlib
import librosa
import pandas as pd
import torch


PROJ_DIR = pathlib.Path("C:\\Users\\wesle\\source\\repos\\who-said-what")
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

data_folder = PROJ_DIR / "training_data"
files = list(data_folder.glob('*.m4a'))
csv = pd.DataFrame(index=range(len(files)), columns=["location", "label"])

for i, file in enumerate(files):

    # load and transform audio
    audio, _ = librosa.load(file, sr=RATE)
    spectrogram = tfm(audio)

    # save spectrogram tensor
    file_name = file.stem
    save_path = data_folder / pathlib.Path(f"{SPEC_TYPE}_{file_name}.pt")
    torch.save(spectrogram, save_path)

    # update annotations
    person = file_name.split("_")[0]
    csv.loc[i] = [save_path, label_map[person]]

csv.to_csv(data_folder / "annotations.csv", header=False, index=False)









