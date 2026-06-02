import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mainrun"))
import model  # type: ignore  # noqa: E402  (mainrun/ added to sys.path above)


def tiny_cfg(**overrides) -> "model.GPTConfig":
    """Small GPTConfig for fast tests; override fields as needed."""
    base = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32, dropout=0.0, eos_id=1)
    base.update(overrides)
    return model.GPTConfig(**base)


class TestModel(unittest.TestCase):
    def test_forward_shapes_and_loss(self):
        torch.manual_seed(0)
        cfg = tiny_cfg()
        m = model.GPT(cfg).eval()
        idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        logits, loss = m(idx, idx)
        self.assertEqual(tuple(logits.shape), (2, cfg.block_size, cfg.vocab_size))
        self.assertTrue(torch.isfinite(loss))

    def test_no_targets_gives_no_loss(self):
        m = model.GPT(tiny_cfg()).eval()
        _, loss = m(torch.zeros(1, 4, dtype=torch.long))
        self.assertIsNone(loss)

    def test_real_config_param_count(self):
        # Guards the validated invariant: the submitted config has this many params.
        cfg = tiny_cfg(vocab_size=16000, block_size=128, n_layer=6, n_head=8, d_model=512, dropout=0.1)
        params = sum(p.numel() for p in model.GPT(cfg).parameters() if p.requires_grad)
        self.assertEqual(params, 27_107_328)

    def test_residual_init_scaling(self):
        torch.manual_seed(0)
        cfg = tiny_cfg(n_layer=6)
        m = model.GPT(cfg)
        expected = 0.02 / math.sqrt(2 * cfg.n_layer)
        self.assertAlmostEqual(m.blocks[0].attn.proj.weight.std().item(), expected, delta=0.002)
        self.assertAlmostEqual(m.blocks[0].mlp.proj.weight.std().item(), expected, delta=0.002)
        self.assertAlmostEqual(m.blocks[0].attn.qkv.weight.std().item(), 0.02, delta=0.004)

    def test_weight_tying(self):
        m = model.GPT(tiny_cfg())
        self.assertIs(m.head.weight, m.token_emb.weight)

    def test_rotate_half(self):
        x = torch.arange(8.0).view(1, 1, 1, 8)
        r = model.rotate_half(x)
        self.assertTrue(torch.equal(r[..., :4], -x[..., 4:]))
        self.assertTrue(torch.equal(r[..., 4:], x[..., :4]))

    def test_per_title_mask_blocks_cross_title(self):
        # A token in a later title must be unaffected by earlier-title content,
        # since attention is blocked across the <eos> boundary (id 1).
        torch.manual_seed(0)
        m = model.GPT(tiny_cfg(dropout=0.0)).eval()
        a = torch.tensor([[5, 6, 1, 7, 8, 9]])
        b = torch.tensor([[3, 4, 1, 7, 8, 9]])   # differs only in the first title
        with torch.no_grad():
            la, _ = m(a)
            lb, _ = m(b)
        # positions 3..5 belong to the second title -> identical logits
        self.assertTrue(torch.allclose(la[:, 3:, :], lb[:, 3:, :], atol=1e-5))
        # position 0 is in the first title -> must differ
        self.assertFalse(torch.allclose(la[:, 0, :], lb[:, 0, :], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
