import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mainrun"))
import data  # type: ignore  # noqa: E402  (mainrun/ added to sys.path above)

TITLES = ["show hn my project", "ask hn career advice", "rust is fast", "python typing tips"]


def tiny_tokenizer() -> "data.BPETokenizer":
    """A small BPE tokenizer trained on a handful of titles."""
    return data.BPETokenizer(data.train_tokenizer(TITLES, vocab_size=200, eos_token="<eos>"))


class TestData(unittest.TestCase):
    def test_eos_is_single_token(self):
        # The per-title attention mask relies on <eos> encoding atomically.
        tok = tiny_tokenizer()
        eos_id = tok.tk.token_to_id("<eos>")
        ids = tok.encode("hello<eos>world<eos>")
        self.assertEqual(ids.count(eos_id), 2)

    def test_encode_decode_roundtrip(self):
        tok = tiny_tokenizer()
        ids = tok.encode("rust is fast")
        self.assertIsInstance(ids, list)
        self.assertIn("rust", tok.decode(ids))

    def test_get_batch_shapes_and_pointer(self):
        ids = torch.arange(1000)
        x, y, ptr = data.get_batch(ids, 0, block_size=8, batch_size=4, device="cpu")
        self.assertEqual(tuple(x.shape), (4, 8))
        self.assertEqual(tuple(y.shape), (4, 8))
        self.assertEqual(ptr, 8 * 4)
        self.assertTrue(torch.equal(x[0, 1:], y[0, :-1]))  # y is x shifted by one

    def test_get_batch_wraps_at_end(self):
        ids = torch.arange(40)  # span = 8*4+1 = 33; ptr 20 + 33 >= 40 -> wrap to 0
        x, _, _ = data.get_batch(ids, 20, block_size=8, batch_size=4, device="cpu")
        self.assertEqual(x[0, 0].item(), 0)

    def test_iter_full_split_count(self):
        ids = torch.arange(1000)
        span = 8 * 4 + 1
        batches = list(data.iter_full_split(ids, block_size=8, batch_size=4, device="cpu"))
        self.assertEqual(len(batches), (1000 - span) // span + 1)


if __name__ == "__main__":
    unittest.main()
