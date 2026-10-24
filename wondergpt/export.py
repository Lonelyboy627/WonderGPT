"""Export an inference-only checkpoint without optimizer or random-number states."""
import argparse
from pathlib import Path

from .runtime import load_checkpoint, save_checkpoint


def export_checkpoint(source, target):
    state = load_checkpoint(source)
    exported = {name: state[name] for name in ('format_version', 'model_config', 'model', 'data_metadata', 'step')}
    exported['kind'] = 'inference'
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve() == Path(source).resolve():
        raise ValueError('export destination must differ from the training checkpoint')
    save_checkpoint(target, exported)
    return exported


def main():
    parser = argparse.ArgumentParser(description='Export a compact checkpoint for sampling.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        state = export_checkpoint(args.checkpoint, args.output)
        print(f'Exported step {state["step"]} to {args.output} ({args.output.stat().st_size:,} bytes)')
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
