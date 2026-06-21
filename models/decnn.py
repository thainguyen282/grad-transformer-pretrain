import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def _num_doublings(start_size: int, target_size: int) -> int:
    """How many stride-2 doublings to reach >= target_size from start_size."""
    if target_size <= start_size:
        return 0
    return math.ceil(math.log2(target_size / start_size))


class DECNN(nn.Module):
    """
    Args:
        hidden_dim: size of the input hidden/latent vector (flat embedding,
            no spatial structure — shape (hidden_dim,)).
        target_shape: (channels, height, width) of the desired output image.
        base_channels: channel count right after the initial projection.
        start_size: spatial size (S x S) the embedding gets "unfolded" into
            before the doubling stages begin.
        min_channels: channel count won't be halved below this on the way down.
        use_linear_projection: if True (default), use nn.Linear + reshape to
            create the (base_channels, start_size, start_size) seed. If False,
            skip the Linear entirely and instead treat the embedding as a
            (hidden_dim, 1, 1) tensor, using a single
            ConvTranspose2d(kernel_size=start_size, stride=1, padding=0) as the
            first deconv layer to do the same unfolding. The two are
            mathematically equivalent (both are just a learned linear map from
            hidden_dim -> base_channels*start_size*start_size); use
            use_linear_projection=False if you want the model to be "pure
            conv" with no nn.Linear layer, e.g. to match a DCGAN-style
            generator convention.
    """

    def __init__(
        self,
        hidden_dim: int,
        target_shape: tuple[int, int, int],
        base_channels: int = 256,
        start_size: int = 4,
        min_channels: int = 32,
        use_linear_projection: bool = True,
    ):
        super().__init__()
        out_channels, target_h, target_w = target_shape
        self.start_size = start_size
        self.base_channels = base_channels
        self.use_linear_projection = use_linear_projection
        self.hidden_dim = hidden_dim

        n_layers = max(
            _num_doublings(start_size, target_h),
            _num_doublings(start_size, target_w),
            1,  # always at least one upsampling stage
        )

        if use_linear_projection:
            # Embedding -> Linear -> reshape to (base_channels, start_size, start_size)
            self.project = nn.Linear(
                hidden_dim, base_channels * start_size * start_size
            )
            self.first_conv = None
        else:
            # Embedding -> reshape to (hidden_dim, 1, 1) -> single ConvTranspose2d
            # unfolds it straight to (base_channels, start_size, start_size).
            # Equivalent to the Linear above, but stays purely convolutional.
            self.project = None
            self.first_conv = nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dim,
                    base_channels,
                    kernel_size=start_size,
                    stride=1,
                    padding=0,
                ),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
            )

        layers = []
        in_ch = base_channels
        for _ in range(n_layers):
            out_ch = max(in_ch // 2, min_channels)
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        # Snap to the exact requested (H, W) and channel count.
        layers += [
            nn.Upsample(
                size=(target_h, target_w), mode="bilinear", align_corners=False
            ),
            nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        ]

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, hidden_dim) -- a flat embedding, no spatial structure.
        if self.use_linear_projection:
            x = self.project(z)
            x = x.view(-1, self.base_channels, self.start_size, self.start_size)
        else:
            x = z.view(-1, self.hidden_dim, 1, 1)
            x = self.first_conv(x)
        return self.decoder(x)
