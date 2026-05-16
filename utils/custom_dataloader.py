import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def load_sample_from_disk(index, model_pairs, model_load_path, num_noisy_samples):
    subset_size = num_noisy_samples ** 2
    pair_idx    = index // subset_size
    data_pair   = index % subset_size
    small_idx   = data_pair // num_noisy_samples
    large_idx   = data_pair % num_noisy_samples
 
    small_model_path, large_model_path = model_pairs[pair_idx]
 
    x = torch.load(os.path.join(model_load_path, small_model_path, f"sample_{small_idx}.pt"), weights_only=True)
    y = torch.load(os.path.join(model_load_path, large_model_path, f"sample_{large_idx}.pt"), weights_only=True)
 
    return x, y 

def pad_sequence_to_target(x_list, target_h, target_w):
    padded_x = []
    masks = []

    for x in x_list:
        x_pad, mask = pad_to_target(x, target_h, target_w)
        padded_x.append(x_pad)
        masks.append(mask)

    padded_x = torch.stack(padded_x)
    masks = torch.stack(masks)

    return padded_x, masks

def pad_to_target(x, target_h, target_w):
    h, w = x.shape
    if h > target_h or w > target_w:
        raise ValueError(
            f"Matrix size ({h}x{w}) exceeds target size ({target_h}x{target_w})."
        )
    pad_h      = target_h - h
    pad_w      = target_w - w
    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left

    x = x.unsqueeze(0)  # [H, W] → [1, H, W]
    x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0)
 
    mask = torch.zeros(1, target_h, target_w, dtype=x.dtype)
    mask[:, pad_top:pad_top + h, pad_left:pad_left + w] = 1.0
 
    return x_pad, mask
class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, indices, model_pairs, model_load_path,
                 num_noisy_samples, target_h=3584, target_w=3584):

        self.indices = indices
        self.model_pairs = model_pairs
        self.model_load_path = model_load_path
        self.num_noisy_samples = num_noisy_samples
        self.target_h = target_h
        self.target_w = target_w

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos):
        print("reach here")
        idx = self.indices[pos]

        subset_size = self.num_noisy_samples ** 2
        pair_idx = idx // subset_size
        data_pair = idx % subset_size

        small_idx = data_pair // self.num_noisy_samples
        large_idx = data_pair % self.num_noisy_samples

        small_model_path, large_model_path = self.model_pairs[pair_idx]


        x_list = torch.load(
            os.path.join(
                self.model_load_path,
                small_model_path,
                f"sample_{small_idx}.pt"
            ),
            weights_only=True
        )

        y_list = torch.load(
            os.path.join(
                self.model_load_path,
                large_model_path,
                f"sample_{large_idx}.pt"
            ),
            weights_only=True
        )
        print("loading data")
        x_list, x_masks = pad_sequence_to_target(x_list, self.target_h, self.target_w)
        y_list, y_masks = pad_sequence_to_target(y_list, self.target_h, self.target_w)  
        x_list, x_masks, y_list, y_masks = x_list.to(dtype=torch.bfloat16), x_masks.to(dtype=torch.bfloat16), y_list.to(dtype=torch.bfloat16), y_masks.to(dtype=torch.bfloat16)

        return x_list, y_list, x_masks, y_masks

def collate_fn(batch):
    # batch = [(x_list, y_list), ...]
    return batch


def build_dataloader(indices, model_pairs, model_load_path, num_noisy_samples,
                     target_h=3584, target_w=3584,
                     batch_size=32, shuffle=True, num_workers=0):

    dataset = IndexDataset(
        indices,
        model_pairs,
        model_load_path,
        num_noisy_samples,
        target_h,
        target_w,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )