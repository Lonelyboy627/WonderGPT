import unittest

import torch

from wondergpt.model import GPT, GPTConfig


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = GPT(GPTConfig(vocab_size=8, block_size=8, n_layer=2,
                                   n_head=2, n_embd=16, dropout=0.0))

    def test_causal_attention_cannot_see_future_tokens(self):
        self.model.eval()
        first = torch.tensor([[0, 1, 2, 3, 4, 5]])
        second = torch.tensor([[0, 1, 2, 7, 6, 0]])
        with torch.no_grad():
            left, _ = self.model(first)
            right, _ = self.model(second)
        torch.testing.assert_close(left[:, :3], right[:, :3], rtol=0, atol=1e-6)
        self.assertFalse(torch.allclose(left[:, 3:], right[:, 3:]))

    def test_shapes_gradients_and_weight_tying(self):
        x = torch.randint(8, (2, 8))
        logits, loss = self.model(x, x.roll(-1, dims=1))
        self.assertEqual(logits.shape, (2, 8, 8))
        self.assertIs(self.model.lm_head.weight, self.model.token_embedding.weight)
        loss.backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_small_model_can_learn_next_token_pattern(self):
        x = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]).repeat(4, 1)
        y = (x + 1) % 4
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.02)
        initial = self.model(x, y)[1].item()
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            _, loss = self.model(x, y)
            loss.backward()
            optimizer.step()
        self.assertLess(self.model(x, y)[1].item(), initial * 0.2)

    def test_generation_crops_context_and_is_seeded(self):
        prompt = torch.tensor([[0, 1] * 8])
        self.model.train()
        torch.manual_seed(123)
        a = self.model.generate(prompt, 12, top_k=3)
        torch.manual_seed(123)
        b = self.model.generate(prompt, 12, top_k=3)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a[:, :16], prompt))
        self.assertEqual(a.shape, (1, 28))
        self.assertTrue(self.model.training)

    def test_invalid_configuration_and_sampling(self):
        with self.assertRaises(ValueError):
            GPTConfig(vocab_size=8, n_head=3, n_embd=16)
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1, 9), dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model.generate(torch.zeros((1, 1), dtype=torch.long), 1, temperature=0)
        with self.assertRaises(ValueError):
            self.model.generate(torch.zeros((1, 1), dtype=torch.long), 1, top_k=-1)


if __name__ == '__main__':
    unittest.main()
