import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=img_dim, out_features=128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers.forward(x)


class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers.forward(x)
