import torch
from torch import optim, nn
import torch.nn.functional as func
import numpy as np

import matplotlib.pyplot as plt


class AVNN(nn.Module):
    """
    The AVNN (audio to visual NN) model is a CNN that takes a tensor containing 3 channels.
    Each of 3 channels is a different visual repreentation of the original audio data:
    spectrogram, mel-spectrogram, and fingerprint spectrogram.

    The model outputs the probabilities of the number of speakers being 1, 2, or 3+
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 256, 5)
        self.conv4 = nn.Conv2d(256, 512, 3)
        self.fc1 = nn.Linear(512 * 25 * 51, 1028)
        self.fc2 = nn.Linear(1028, 512)
        self.out = nn.Linear(512, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = self.drop(self.pool(x))
        x = func.relu(self.conv2(x))
        x = self.drop(self.pool(x))
        x = func.relu(self.conv3(x))
        x = self.drop(self.pool(x))
        x = func.relu(self.conv4(x))
        x = x.view(-1, 512 * 25 * 51)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        return self.out(x)


def train_(model, epochs, lr, trainloader, validloader=None, plot=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loss = []
    valid_loss = []
    accuracy = []

    for e in range(epochs):

        running_tl = 0
        running_vl = 0
        running_ac = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            scores = model(images)
            t_loss = criterion(scores, labels)
            running_tl += t_loss.item()

            t_loss.backward()
            opt.step()

        model.train_loss.append(running_tl / len(trainloader))

        if validloader is not None:
            model.eval()
            with torch.no_grad():
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    scores = model(images)
                    ps = func.log_softmax(scores, dim=1)
                    preds = torch.argmax(ps, dim=1)

                    v_loss = criterion(scores, labels)
                    running_vl += v_loss.item()
                    running_ac += (preds == labels).cpu().numpy().mean()
            model.valid_loss.append(running_vl / len(validloader))
            model.accuracy.apend(running_ac / len(validloader))
            model.train()

        if plot:
            fig, axes = plt.subplots(1, 2)
            axes[0].title('Loss Plot')
            axes[1].title('Accuracy')
            axes[0].plot(train_loss, label='training')
            axes[0].plot(valid_loss, label='validation')
            axes[1].plot(accuracy)
            axes[0].xlabel('Epochs')
            axes[1].xlabel('Epochs')
            plt.legend()
            plt.show()




if __name__ == '__main__':
    network = AVNN()
    im = torch.tensor(np.random.random((3, 257, 460)))
    im = im.unsqueeze(0).float()
    out = network(im)
    print(im.shape)
    print(out.shape)
