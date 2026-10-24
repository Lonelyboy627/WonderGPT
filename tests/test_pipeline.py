from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from wondergpt.data import CharTokenizer, Dataset, prepare
from wondergpt.export import export_checkpoint
from wondergpt.model import GPT, GPTConfig
from wondergpt.runtime import load_checkpoint
from wondergpt.train import TrainConfig, learning_rate_at, run


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / 'data'
        prepare('Once upon a time.\n' * 100, 'A prince came.\n' * 50, self.data)

    def tearDown(self):
        self.temporary.cleanup()

    def test_unicode_tokenizer_and_unknown_prompt(self):
        text = '“fairy” café\n'
        tokenizer = CharTokenizer(sorted(set(text)))
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        with self.assertRaisesRegex(ValueError, 'absent'):
            tokenizer.encode('中文')

    def test_binary_integrity_and_shifted_targets(self):
        dataset = Dataset(self.data)
        x, y = dataset.batch('train', 4, 16, 'cpu', torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
        path = self.data / 'train.bin'
        path.write_bytes(path.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, 'does not match'):
            Dataset(self.data)

    def test_learning_rate_boundaries(self):
        config = TrainConfig(warmup_iters=2, lr_decay_iters=10)
        self.assertAlmostEqual(learning_rate_at(0, config), config.learning_rate / 2)
        self.assertAlmostEqual(learning_rate_at(2, config), config.learning_rate)
        self.assertAlmostEqual(learning_rate_at(10, config), config.min_lr)
        self.assertAlmostEqual(learning_rate_at(20, config), config.min_lr)

    def test_resume_matches_uninterrupted_cpu_training_and_generates(self):
        options = dict(block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.2)
        config = TrainConfig(data_dir=str(self.data), out_dir=str(self.root / 'split'),
                             device='cpu', batch_size=2, max_iters=2, eval_interval=2,
                             eval_iters=1, log_interval=100, threads=1, amp=False)
        run(options, config)
        run(options, replace(config, max_iters=4), self.root / 'split/latest.pt')
        run(options, replace(config, max_iters=4, out_dir=str(self.root / 'whole')))
        split = load_checkpoint(self.root / 'split/latest.pt')
        whole = load_checkpoint(self.root / 'whole/latest.pt')
        self.assertEqual(split['step'], 4)
        for name in split['model']:
            torch.testing.assert_close(split['model'][name], whole['model'][name], rtol=0, atol=0)
        tokenizer = CharTokenizer(split['data_metadata']['chars'])
        model = GPT(GPTConfig(**split['model_config']))
        model.load_state_dict(split['model'])
        prompt = torch.tensor([tokenizer.encode('Once')])
        text = tokenizer.decode(model.generate(prompt, 20, top_k=1)[0].tolist())
        self.assertEqual(len(text), 24)
        self.assertTrue(text.startswith('Once'))
        self.assertTrue((self.root / 'split/best.pt').is_file())
        exported_path = self.root / 'inference.pt'
        exported = export_checkpoint(self.root / 'split/latest.pt', exported_path)
        self.assertNotIn('optimizer', exported)
        self.assertNotIn('torch_rng_state', exported)
        exported_model = GPT(GPTConfig(**exported['model_config']))
        exported_model.load_state_dict(load_checkpoint(exported_path)['model'])
        with self.assertRaisesRegex(ValueError, 'inference exports'):
            run(options, replace(config, max_iters=5), exported_path)
        metrics = [json.loads(line) for line in (self.root / 'split/metrics.jsonl').read_text().splitlines()]
        self.assertEqual([entry['step'] for entry in metrics], [0, 2, 4])

    def test_resume_rejects_changed_data_and_existing_run_is_protected(self):
        options = dict(block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
        config = TrainConfig(data_dir=str(self.data), out_dir=str(self.root / 'run'),
                             device='cpu', batch_size=2, max_iters=1, eval_iters=1, threads=1)
        run(options, config)
        with self.assertRaisesRegex(ValueError, 'contains a run'):
            run(options, config)
        prepare('a different text\n' * 100, 'different validation\n' * 50, self.data)
        with self.assertRaisesRegex(ValueError, 'resume'):
            run(options, replace(config, max_iters=2), self.root / 'run/latest.pt')


if __name__ == '__main__':
    unittest.main()
