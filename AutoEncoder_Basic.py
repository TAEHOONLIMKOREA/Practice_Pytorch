import torch
from torch import nn

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_features=28 * 28),
            nn.Linear(28 * 28, 2),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(2, 28 * 28),
            nn.BatchNorm1d(num_features=28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x
