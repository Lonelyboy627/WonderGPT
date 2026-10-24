"""Portable uint16 token files and a reversible character tokenizer."""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


class CharTokenizer:
    def __init__(self, chars):
        if not chars or len(chars) != len(set(chars)) or any(len(ch) != 1 for ch in chars):
            raise ValueError('vocabulary must contain unique single characters')
        if len(chars) > 65536:
            raise ValueError('uint16 supports at most 65536 characters')
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}

    def encode(self, text):
        unknown = sorted(set(text) - self.stoi.keys())
        if unknown:
            raise ValueError(f'Characters absent from the training vocabulary: {unknown!r}')
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return ''.join(self.chars[i] for i in ids)


def prepare(train_text, val_text, out_dir):
    if not train_text or not val_text:
        raise ValueError('training and validation text must both be nonempty')
    tokenizer = CharTokenizer(sorted(set(train_text + val_text)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(format_version=1, dtype='uint16', byte_order='little', chars=tokenizer.chars, splits={})
    for name, text in [('train', train_text), ('val', val_text)]:
        tokens = array('H', tokenizer.encode(text))
        if sys.byteorder != 'little':
            tokens.byteswap()
        raw = tokens.tobytes()
        (out_dir / f'{name}.bin').write_bytes(raw)
        metadata['splits'][name] = dict(tokens=len(text), sha256=sha256(raw),
                                        source_text_sha256=sha256(text.encode('utf-8')))
    (out_dir / 'meta.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return metadata


class Dataset:
    def __init__(self, path):
        path = Path(path)
        self.metadata = json.loads((path / 'meta.json').read_text(encoding='utf-8'))
        if (self.metadata.get('format_version'), self.metadata.get('dtype'), self.metadata.get('byte_order')) != (1, 'uint16', 'little'):
            raise ValueError('unsupported token data format; run prepare.py again')
        self.tokenizer = CharTokenizer(self.metadata['chars'])
        self.splits = {}
        for name in ('train', 'val'):
            raw = (path / f'{name}.bin').read_bytes()
            entry = self.metadata['splits'][name]
            if len(raw) != entry['tokens'] * 2 or sha256(raw) != entry['sha256']:
                raise ValueError(f'{name}.bin does not match meta.json; run prepare.py again')
            tokens = array('H')
            tokens.frombytes(raw)
            if sys.byteorder != 'little':
                tokens.byteswap()
            tensor = torch.tensor(tokens, dtype=torch.long)
            if tensor.numel() == 0 or tensor.max().item() >= len(self.tokenizer.chars):
                raise ValueError(f'{name}.bin has invalid token IDs')
            self.splits[name] = tensor

    def batch(self, split, batch_size, block_size, device, generator):
        tokens = self.splits[split]
        if len(tokens) <= block_size:
            raise ValueError(f'{split} requires more than {block_size} characters')
        starts = torch.randint(len(tokens) - block_size, (batch_size,), generator=generator)
        positions = starts[:, None] + torch.arange(block_size)
        return tokens[positions].to(device), tokens[positions + 1].to(device)


def main():
    parser = argparse.ArgumentParser(description='Prepare character-level token files for WonderGPT.')
    parser.add_argument('--train-file', type=Path)
    parser.add_argument('--val-file', type=Path)
    parser.add_argument('--input-file', type=Path, help='Alternative: split one UTF-8 file by position (90/10).')
    parser.add_argument('--out-dir', type=Path, default=Path('data/fantasy'))
    args = parser.parse_args()
    if args.input_file:
        if args.train_file or args.val_file:
            parser.error('--input-file cannot be combined with explicit split files')
        text = args.input_file.read_text(encoding='utf-8')
        boundary = int(0.9 * len(text))
        train_text, val_text = text[:boundary], text[boundary:]
    else:
        if bool(args.train_file) != bool(args.val_file):
            parser.error('--train-file and --val-file must be provided together')
        corpus = PROJECT_ROOT / 'public_domain_fantasy_en'
        train_text = (args.train_file or corpus / 'train.txt').read_text(encoding='utf-8')
        val_text = (args.val_file or corpus / 'val.txt').read_text(encoding='utf-8')
    metadata = prepare(train_text, val_text, args.out_dir)
    print(f'Vocabulary: {len(metadata["chars"])} characters')
    print(f'Train: {len(train_text):,} tokens; validation: {len(val_text):,} tokens')
    print(f'Written to {args.out_dir.resolve()}')


if __name__ == '__main__':
    main()
