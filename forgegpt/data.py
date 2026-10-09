import os
import numpy as np
import torch

_files = {}

def get_batch(split, data_dir, block_size, batch_size, device):
    """Pick random snippets. For each, y is x shifted one token ahead (the 'answer')."""
    path = os.path.join(data_dir, f"{split}.bin")
    if path not in _files:
        _files[path] = np.memmap(path, dtype=np.uint16, mode="r")
    data = _files[path]

    starts = torch.randint(len(data) - block_size - 1, (batch_size,)).tolist()
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in starts])
    return x.to(device), y.to(device)
