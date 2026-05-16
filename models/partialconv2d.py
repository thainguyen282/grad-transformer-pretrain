###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        print(f"[forward] START — input.shape={input.shape}, mask_in={'provided' if mask_in is not None else 'None'}")
        assert len(input.shape) == 4
        
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)
            print(f"[forward] Entered mask/size update block — last_size set to {self.last_size}")

            with torch.no_grad():
                print(f"[forward] weight_maskUpdater.type={self.weight_maskUpdater.type()}, input.type={input.type()}")
                if self.weight_maskUpdater.type() != input.type():
                    print("[forward] Casting weight_maskUpdater to input type")
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    print(f"[forward] No mask_in — creating default mask. multi_channel={self.multi_channel}")
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                    print(f"[forward] Created mask.shape={mask.shape}")
                else:
                    mask = mask_in
                    print(f"[forward] Using provided mask_in.shape={mask.shape}")

                print(f"[forward] Running F.conv2d for update_mask — stride={self.stride}, padding={self.padding}, dilation={self.dilation}")
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
                print(f"[forward] update_mask.shape={self.update_mask.shape}, min={self.update_mask.min():.4f}, max={self.update_mask.max():.4f}")

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                print(f"[forward] mask_ratio computed — slide_winsize={self.slide_winsize}, mask_ratio stats: min={self.mask_ratio.min():.4f}, max={self.mask_ratio.max():.4f}")

                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)
                print(f"[forward] After clamp+mul — update_mask.shape={self.update_mask.shape}, mask_ratio.shape={self.mask_ratio.shape}")

        print(f"[forward] Calling super().forward() — input multiplied by mask: {mask_in is not None}")
        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)
        print(f"[forward] raw_out.shape={raw_out.shape}, min={raw_out.min():.4f}, max={raw_out.max():.4f}")

        if self.bias is not None:
            print(f"[forward] Applying bias — bias.shape={self.bias.shape}, out_channels={self.out_channels}")
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            print("[forward] No bias — applying mask_ratio directly")
            output = torch.mul(raw_out, self.mask_ratio)

        print(f"[forward] output.shape={output.shape}, min={output.min():.4f}, max={output.max():.4f}")
        print(f"[forward] END — return_mask={self.return_mask}")

        if self.return_mask:
            return output, self.update_mask
        else:
            return output