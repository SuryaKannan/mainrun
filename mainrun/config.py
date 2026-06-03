from dataclasses import dataclass, fields


@dataclass
class Hyperparameters:
    block_size: int = 128
    batch_size: int = 64
    vocab_size: int = 16_000
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 512
    dropout: float = 0.1
    lr: float = 1.5e-3
    weight_decay: float = 0.1
    warmup_frac: float = 0.6
    evals_per_epoch: int = 3

    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"


def load_hyperparameters(config_path: str | None = None) -> Hyperparameters:
    """Build Hyperparameters from defaults, optionally overridden by a YAML file."""
    args = Hyperparameters()
    if config_path:
        import yaml
        with open(config_path) as f:
            overrides = yaml.safe_load(f) or {}
        valid = {field.name for field in fields(args)}
        for key, value in overrides.items():
            if key not in valid:
                raise ValueError(f"unknown hyperparameter '{key}' in {config_path}")
            setattr(args, key, value)
    return args