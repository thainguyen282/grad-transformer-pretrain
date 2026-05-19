import torch
import torch.nn as nn
import torch.nn.functional as F


class DECNN(nn.Module):
    def __init__(self, latent_dim=2048, output_size=2048):
        super(DECNN, self).__init__()
        self.output_size = output_size

        # 1) vector → feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 4 * 4),
            nn.LayerNorm(1024 * 4 * 4),
            nn.ReLU(inplace=True)
        )

        # 2) upsampling backbone (hidden inside one class)
        def block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        decoder_blocks = []
        in_channels = 1024
        channels = [1024, 512, 256, 128, 64, 32, 16, 8, 4]
        spatial_size = 4
        while spatial_size < output_size:
            out_channels = channels[min(len(decoder_blocks), len(channels) - 1)]
            decoder_blocks.append(block(in_channels, out_channels))
            in_channels = out_channels
            spatial_size *= 2

        self.decoder = nn.Sequential(*decoder_blocks)

        # 3) output head
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        if x.dim() < 2:
            raise ValueError(f"Expected latent tensor with shape [..., D], got {x.shape}")

        leading_shape = x.shape[:-1]
        latent_dim = x.shape[-1]
        x = x.reshape(-1, latent_dim)

        x = self.fc(x)                     # (N, 1024*4*4)
        x = x.view(-1, 1024, 4, 4)         # reshape
        x = self.decoder(x)                # upsampling stack
        if x.size(-1) != self.output_size or x.size(-2) != self.output_size:
            x = F.interpolate(x, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        x = self.out(x)                    # grayscale output
        x = x.squeeze(1)
        x = x.reshape(*leading_shape, *x.shape[1:])
        return x