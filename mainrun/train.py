import utils  # noqa: F401, checks if we are in devcontainer
import random
import time
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from tqdm import tqdm

from logging_utils import configure_logging
from data import get_titles, get_batch, iter_full_split, train_tokenizer, BPETokenizer
from model import GPTConfig, GPT


@dataclass
class Hyperparameters:
    block_size: int = 128
    batch_size: int = 64
    vocab_size: int = 16_000
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 512
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_frac: float = 0.1
    evals_per_epoch: int = 3

    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"


logger = None


def main() -> None:
    """Initialise training loop on Hacker News titles, logging metrics to the configured file."""
    args = Hyperparameters()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    global logger
    logger = configure_logging(args.log_file)

    hyperparams_dict = vars(args)
    logger.log("hyperparameters_configured", **hyperparams_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("device_info", device=device)

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)

    eos_token = "<eos>"
    tok = BPETokenizer(train_tokenizer(train_titles+val_titles, args.vocab_size, eos_token=eos_token))
    train_text = eos_token.join(train_titles) + eos_token
    val_text = eos_token.join(val_titles) + eos_token
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    eos_id = tok.tk.token_to_id(eos_token)

    batches = len(train_ids) // (args.block_size * args.batch_size)
    max_steps = args.epochs * batches
    eval_interval = batches // args.evals_per_epoch
    logger.log("dataset_info",
               titles_count=len(train_titles),
               epochs=args.epochs,
               batches_per_epoch=batches,
               tokens_per_epoch=len(train_ids),
               vocab_size=tok.vocab_size)

    cfg = GPTConfig(
        vocab_size = tok.vocab_size,
        block_size = args.block_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        d_model    = args.d_model,
        dropout    = args.dropout,
        eos_id     = eos_id,
    )
    model = GPT(cfg).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    warmup_steps = max(1, int(max_steps * args.warmup_frac))
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0 / warmup_steps, end_factor=1.0, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps - warmup_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])

    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        return losses / len(val_text)

    ptr = 0
    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
            step += 1
            xb, yb, ptr = get_batch(train_ids, ptr, args.block_size, args.batch_size, device)
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            elapsed = time.time() - t0
            logger.log("training_step",
                      step=step,
                      max_steps=max_steps,
                      loss=loss.item(),
                      elapsed_time=elapsed,
                      prnt=False)

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate()
                logger.log("validation_step",
                          step=step,
                          max_steps=max_steps,
                          loss=val_loss,
                          elapsed_time=elapsed)


if __name__ == "__main__":
    try:
        main()
    finally:
        if logger and hasattr(logger, 'file_handler'):
            logger.file_handler.close()
