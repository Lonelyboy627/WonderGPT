"""Sample text from a saved GPT without loading the training corpus."""
import argparse
from pathlib import Path

import torch

from .data import CharTokenizer
from .model import GPT, GPTConfig
from .runtime import load_checkpoint, resolve_device


def main():
    parser = argparse.ArgumentParser(description='Generate English fantasy text with a trained GPT.')
    parser.add_argument('--checkpoint', type=Path, default=Path('out/tiny/best.pt'))
    parser.add_argument('--prompt', default='Once upon a time')
    parser.add_argument('--max-new-tokens', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=40, help='0 disables top-k filtering.')
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.num_samples <= 0 or args.top_k < 0:
            raise ValueError('num-samples must be positive and top-k must be nonnegative')
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        device = resolve_device(args.device)
        state = load_checkpoint(args.checkpoint)
        tokenizer = CharTokenizer(state['data_metadata']['chars'])
        model = GPT(GPTConfig(**state['model_config'])).to(device)
        model.load_state_dict(state['model'])
        model.eval()
        prompt = args.prompt or '\n'
        encoded = tokenizer.encode(prompt)
        samples = []
        for _ in range(args.num_samples):
            context = torch.tensor([encoded], dtype=torch.long, device=device)
            tokens = model.generate(context, args.max_new_tokens, args.temperature, args.top_k or None)
            samples.append(tokenizer.decode(tokens[0].tolist()))
        text = '\n\n---\n\n'.join(samples)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + '\n', encoding='utf-8')
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
