import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(
                channels_img,
                features_d,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 2, 2, 1),
            self._block(features_d*2, features_d*4, 2, 2, 1),
            self._block(features_d*4, features_d*8, 2, 2, 1),
            nn.Conv2d(features_d*8, 1, kernel_size=2, stride=2, padding=0),
            nn.Flatten(start_dim=1),
            nn.Linear(32,1),
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)