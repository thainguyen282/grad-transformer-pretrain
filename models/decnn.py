import torch
import torch.nn as nn


class DECNN(nn.Module):
    def __init__(self, latent_dim=2048):
        super(DECNN, self).__init__()

        # 1) vector → feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 4 * 4),
            nn.BatchNorm1d(1024 * 4 * 4),
            nn.ReLU(inplace=True)
        )

        # 2) upsampling backbone (hidden inside one class)
        def block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.decoder = nn.Sequential(
            block(1024, 1024),  # 8x8
            block(1024, 512),   # 16x16
            block(512, 256),    # 32x32
            block(256, 128),    # 64x64
            block(128, 64),     # 128x128
            block(64, 32),      # 256x256
            block(32, 16),      # 512x512
            block(16, 8),       # 1024x1024
            block(8, 4),        # 2048x2048
        )

        # 3) output head
        self.out = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)                      # (B, 1024*4*4)
        x = x.view(-1, 1024, 4, 4)         # reshape
        x = self.decoder(x)                # upsampling stack
        x = self.out(x)                    # grayscale output
        return x