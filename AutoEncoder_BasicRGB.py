import torch
from torch import nn

class encoder(nn.Module):
    def __init__(self, latent_dim=4):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.LeakyReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, latent_dim)  # Latent space size customizable
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.encoder(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class decoder(nn.Module):
    def __init__(self, latent_dim=4):
        super(decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.LeakyReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, 64 * 64 * 3),
            nn.Tanh()
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.decoder(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
