"""
Test script for PDResNet50 with:
1. Same-size batch (mask=None)
2. Variable-size batch (with mask from collate_fn)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- adjust this import to match your project structure ---
from pd_resnet import pdresnet50


# ─────────────────────────────────────────────
# Collate function for variable-size inputs
# ─────────────────────────────────────────────

def collate_fn(batch):
    """
    batch: list of (x, label) where x is [C, H, W], possibly different H/W
    Returns:
        x_batch:    [B, C, max_h, max_w]
        mask_batch: [B, 1, max_h, max_w]  (1=valid, 0=padded)
        labels:     [B]
    """
    xs, labels = zip(*batch)

    max_h = max(x.shape[1] for x in xs)
    max_w = max(x.shape[2] for x in xs)

    padded, masks = [], []
    for x in xs:
        c, h, w = x.shape
        pad_h      = max_h - h
        pad_w      = max_w - w
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left

        x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        mask = torch.zeros(1, max_h, max_w, dtype=x.dtype)
        mask[:, pad_top:pad_top + h, pad_left:pad_left + w] = 1.0

        padded.append(x_pad)
        masks.append(mask)

    return (
        torch.stack(padded),        # [B, C, max_h, max_w]
        torch.stack(masks),         # [B, 1, max_h, max_w]
        torch.tensor(labels),       # [B]
    )


# ─────────────────────────────────────────────
# Dummy datasets
# ─────────────────────────────────────────────

class SameSizeDataset(Dataset):
    """All images are 224x224 — simulates standard fixed-size input."""
    def __init__(self, size=8):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x     = torch.randn(1, 224, 224)
        label = torch.randint(0, 10, (1,)).item()
        return x, label

class VariableSizeDataset(Dataset):
    def __init__(self, size=8):
        self.size = size
        self.shapes = [
            (1, 256, 2048),
            (1, 2048, 2048),
            (1, 256, 2048),
            (1, 256, 2048),
            (1, 256, 2048),
            (1, 256, 2048),
            (1, 256, 2048),
            (1, 256, 256),
        ]
        # compute dataset-level max once
        self.max_h = max(s[1] for s in self.shapes)
        self.max_w = max(s[2] for s in self.shapes)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        shape = self.shapes[idx % len(self.shapes)]
        x     = torch.randn(*shape)
        label = torch.randint(0, 10, (1,)).item()

        # pad to dataset max here, in __getitem__
        c, h, w    = x.shape
        pad_h      = self.max_h - h
        pad_w      = self.max_w - w
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left

        x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        mask = torch.zeros(1, self.max_h, self.max_w, dtype=x.dtype)
        mask[:, pad_top:pad_top + h, pad_left:pad_left + w] = 1.0

        return x_pad, mask, label

# ─────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────

def test_same_size_batch(model, device):
    print("\n" + "="*50)
    print("TEST 1: Same-size batch (mask=None)")
    print("="*50)

    dataset = SameSizeDataset(size=8)
    loader  = DataLoader(dataset, batch_size=4)

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(loader):
            x = x.to(device)
            # no mask — all inputs same size
            out = model(x, mask=None)
            print(f"  Batch {batch_idx}: x={tuple(x.shape)}  →  out={tuple(out.shape)}")

    print("PASSED ✓")


def test_variable_size_batch(model, device):
    print("\n" + "="*50)
    print("TEST 2: Variable-size batch (with mask)")
    print("="*50)

    dataset = VariableSizeDataset(size=8)
    loader  = DataLoader(dataset, batch_size=4)

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, mask, labels) in enumerate(loader):
            x    = x.to(device)
            mask = mask.to(device)
            out  = model(x, mask=mask)
            print(f"  Batch {batch_idx}: x={tuple(x.shape)}  mask={tuple(mask.shape)}  →  out={tuple(out.shape)}")
            # check mask values are strictly binary
            assert mask.min() >= 0 and mask.max() <= 1, "Mask values out of [0,1] range!"

    print("PASSED ✓")


def test_single_image(model, device):
    print("\n" + "="*50)
    print("TEST 3: Single image forward (batch_size=1)")
    print("="*50)

    x = torch.randn(1, 1, 224, 224).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x, mask=None)
    print(f"  x={tuple(x.shape)}  →  out={tuple(out.shape)}")
    print("PASSED ✓")


def test_output_shape(model, device):
    print("\n" + "="*50)
    print("TEST 4: Output shape check")
    print("="*50)

    batch_size = 4
    x = torch.randn(batch_size, 1, 224, 224).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x, mask=None)

    expected = (batch_size, 2048)   # ResNet50 → 512 * Bottleneck.expansion(4) = 2048
    assert out.shape == torch.Size(expected), \
        f"Expected output shape {expected}, got {tuple(out.shape)}"
    print(f"  Output shape: {tuple(out.shape)} == {expected}")
    print("PASSED ✓")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = pdresnet50(pretrained=False).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    test_single_image(model, device)
    test_output_shape(model, device)
    test_same_size_batch(model, device)
    test_variable_size_batch(model, device)

    print("\n" + "="*50)
    print("ALL TESTS PASSED ✓")
    print("="*50)