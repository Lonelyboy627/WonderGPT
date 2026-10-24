"""Device selection and atomic checkpoint writes shared by the command-line tools."""
from pathlib import Path
import os

import torch


def resolve_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name not in ('cpu', 'cuda'):
        raise ValueError('device must be auto, cpu, or cuda')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; install a CUDA-enabled PyTorch build or use --device cpu')
    return torch.device(name)


def save_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1:
        raise ValueError('unsupported WonderGPT checkpoint format')
    return checkpoint
