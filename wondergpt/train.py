"""Train a GPT on next-character prediction, with evaluation and resumable state."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
import tomllib

import torch

from .data import Dataset
from .model import GPT, GPTConfig
from .runtime import load_checkpoint, resolve_device, save_checkpoint


@dataclass
class TrainConfig:
    data_dir: str = 'data/fantasy'
    out_dir: str = 'out/tiny'
    device: str = 'auto'
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_iters: int = 2000
    eval_interval: int = 100
    eval_iters: int = 10
    log_interval: int = 10
    learning_rate: float = 0.0003
    min_lr: float = 0.00003
    warmup_iters: int = 20
    lr_decay_iters: int = 2000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1337
    threads: int = 4
    amp: bool = True

    def validate(self):
        for name in ('batch_size', 'gradient_accumulation_steps', 'max_iters', 'eval_interval',
                     'eval_iters', 'log_interval', 'lr_decay_iters', 'threads'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.warmup_iters < 0 or self.lr_decay_iters <= self.warmup_iters:
            raise ValueError('lr_decay_iters must exceed nonnegative warmup_iters')
        if not 0 <= self.min_lr <= self.learning_rate or self.learning_rate <= 0:
            raise ValueError('require 0 <= min_lr <= learning_rate and learning_rate > 0')
        if self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError('grad_clip must be positive and weight_decay must be nonnegative')


def learning_rate_at(step, config):
    if step < config.warmup_iters:
        return config.learning_rate * (step + 1) / config.warmup_iters
    progress = min(1.0, (step - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters))
    return config.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (config.learning_rate - config.min_lr)


def precision_context(device, enabled):
    if device.type != 'cuda' or not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast('cuda', dtype=dtype)


@torch.no_grad()
def estimate_loss(model, dataset, config, device):
    was_training = model.training
    model.eval()
    result = {}
    # Fixed evaluation windows make scores comparable without consuming the training RNG.
    generator = torch.Generator().manual_seed(config.seed + 10000)
    try:
        for split in ('train', 'val'):
            total = 0.0
            for _ in range(config.eval_iters):
                x, y = dataset.batch(split, config.batch_size, model.config.block_size, device, generator)
                with precision_context(device, config.amp):
                    _, loss = model(x, y)
                total += loss.item()
            result[split] = total / config.eval_iters
    finally:
        model.train(was_training)
    return result


def run(model_options, config, resume=None):
    config.validate()
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config.seed)
    dataset = Dataset(config.data_dir)
    model_options = dict(model_options)
    model_options['vocab_size'] = len(dataset.tokenizer.chars)
    model_config = GPTConfig(**model_options)
    if any(len(tokens) <= model_config.block_size for tokens in dataset.splits.values()):
        raise ValueError('both data splits must contain more characters than block_size')
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint:
        if checkpoint.get('kind') == 'inference':
            raise ValueError('inference exports cannot resume training; use the original latest.pt')
        if checkpoint['model_config'] != asdict(model_config):
            raise ValueError('resume requires the same model configuration and vocabulary size')
        if checkpoint['data_metadata'] != dataset.metadata:
            raise ValueError('resume data/vocabulary differs from checkpoint; restore the original prepared data')
        for name in ('batch_size', 'gradient_accumulation_steps', 'learning_rate', 'min_lr',
                     'warmup_iters', 'lr_decay_iters', 'weight_decay', 'grad_clip', 'seed', 'amp'):
            if checkpoint['train_config'][name] != getattr(config, name):
                raise ValueError(f'resume requires the same {name}; use the original training configuration')
    model = GPT(model_config).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': config.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}], lr=config.learning_rate, betas=(0.9, 0.95))
    use_scaler = device.type == 'cuda' and config.amp and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    generator = torch.Generator().manual_seed(config.seed)
    step, best_val = 0, float('inf')
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scaler']:
            scaler.load_state_dict(checkpoint['scaler'])
        step, best_val = checkpoint['step'], checkpoint['best_val_loss']
        generator.set_state(checkpoint['batch_rng_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        if device.type == 'cuda' and checkpoint['cuda_rng_states']:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_states'])
        print(f'Resumed at optimizer step {step}')
    if step >= config.max_iters:
        raise ValueError(f'--max-iters is the total target; it must exceed saved step {step}')
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not resume and ((out / 'latest.pt').exists() or (out / 'metrics.jsonl').exists()):
        raise ValueError(f'{out} contains a run already; use --resume or a new --out-dir')
    (out / 'config.json').write_text(json.dumps(
        dict(model=asdict(model_config), train=asdict(config)), indent=2) + '\n', encoding='utf-8')
    parameters = sum(p.numel() for p in model.parameters())
    print(f'{parameters:,} parameters | {device} | context {model_config.block_size} | vocab {model_config.vocab_size}')
    started = time.perf_counter()
    model.train()

    def evaluate_and_save(current_step):
        nonlocal best_val
        losses = estimate_loss(model, dataset, config, device)
        if not all(math.isfinite(value) for value in losses.values()):
            raise RuntimeError('non-finite evaluation loss; reduce learning rate or disable AMP')
        improved = losses['val'] < best_val
        best_val = min(best_val, losses['val'])
        record = dict(step=current_step, train_loss=losses['train'], val_loss=losses['val'],
                      elapsed_seconds=time.perf_counter() - started, parameters=parameters,
                      device=str(device), tokens_per_step=config.batch_size * model_config.block_size * config.gradient_accumulation_steps)
        with (out / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        state = dict(format_version=1, model_config=asdict(model_config), train_config=asdict(config),
                     model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                     step=current_step, best_val_loss=best_val, data_metadata=dataset.metadata,
                     torch_rng_state=torch.get_rng_state(), batch_rng_state=generator.get_state(),
                     cuda_rng_states=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        save_checkpoint(out / 'latest.pt', state)
        if improved:
            save_checkpoint(out / 'best.pt', state)
        print(f'step {current_step}: train {losses["train"]:.4f}, val {losses["val"]:.4f}', flush=True)
        return record

    if step == 0:
        evaluate_and_save(0)
    last_record = None
    while step < config.max_iters:
        rate = learning_rate_at(step, config)
        for group in optimizer.param_groups:
            group['lr'] = rate
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            x, y = dataset.batch('train', config.batch_size, model_config.block_size, device, generator)
            with precision_context(device, config.amp):
                _, loss = model(x, y)
                scaled_loss = loss / config.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss; reduce learning rate or disable AMP')
            total_loss += loss.item() / config.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if step % config.log_interval == 0:
            print(f'iter {step}: loss {total_loss:.4f}, lr {rate:.6g}', flush=True)
        if step % config.eval_interval == 0 or step == config.max_iters:
            last_record = evaluate_and_save(step)
    return last_record


def main():
    parser = argparse.ArgumentParser(description='Train a character-level GPT Transformer.')
    parser.add_argument('--config', type=Path, default=Path('configs/tiny.toml'))
    parser.add_argument('--data-dir')
    parser.add_argument('--out-dir')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--max-iters', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--eval-interval', type=int)
    parser.add_argument('--resume', type=Path, help='Path to latest.pt; max-iters is the total target step.')
    args = parser.parse_args()
    options = tomllib.loads(args.config.read_text(encoding='utf-8'))
    if set(options) - {'model', 'train'}:
        parser.error('configuration must contain only [model] and [train] sections')
    train_options = options.get('train', {})
    for name in ('data_dir', 'out_dir', 'device', 'max_iters', 'batch_size', 'eval_interval'):
        if getattr(args, name) is not None:
            train_options[name] = getattr(args, name)
    try:
        run(options.get('model', {}), TrainConfig(**train_options), args.resume)
    except (ValueError, FileNotFoundError, TypeError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
