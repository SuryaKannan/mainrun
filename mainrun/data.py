from collections.abc import Iterator

import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def get_titles(num_titles: int, seed: int, val_frac: float) -> tuple[list[str], list[str]]:
    """Load and shuffle Hacker News titles, split into (train, val) lists."""
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data").shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]  # type: ignore  (datasets stubs)
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]


def get_batch(split_ids: torch.Tensor, ptr: int, block_size: int, batch_size: int,
              device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Slice one (x, y) batch from a token stream, wrapping ptr at the end.

    Returns the inputs, the next-token targets, and the advanced pointer.
    """
    span = block_size * batch_size + 1
    if ptr + span >= len(split_ids):
        ptr = 0
    batch = split_ids[ptr: ptr + span]
    x = batch[:-1].view(batch_size, block_size).to(device)
    y = batch[1:].view(batch_size, block_size).to(device)
    return x, y, ptr + block_size * batch_size


def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int,
                    device: str | torch.device) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield every full (x, y) batch over a token stream (drops the remainder)."""
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y


def train_tokenizer(titles: list[str], vocab_size: int, unk_token: str = "<unk>",
                    pad_token: str = "<pad>", eos_token: str = "<eos>") -> Tokenizer:
    """Train a byte-level BPE tokenizer on the titles with the given special tokens."""
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()  # type: ignore  (tokenizers stubs)
    tokenizer.decoder = decoders.ByteLevel()  # type: ignore  (tokenizers stubs)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[pad_token, eos_token, unk_token])  # type: ignore  (tokenizers stubs)
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer


class BPETokenizer:
    """Thin wrapper around a trained tokenizer exposing encode/decode by id."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tk = tokenizer

    def encode(self, s: str) -> list[int]:
        """Encode a string into a list of token ids."""
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back into a string, skipping special tokens."""
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return self.tk.get_vocab_size()
