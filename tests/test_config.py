import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mainrun"))
import config  # type: ignore  # noqa: E402  (mainrun/ added to sys.path above)


def write_yaml(text: str) -> str:
    """Write a YAML snippet to a temp file and return its path."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(text)
        return f.name


class TestConfig(unittest.TestCase):
    def test_no_path_gives_defaults(self):
        hp = config.load_hyperparameters()
        self.assertEqual(hp.lr, 1e-3)
        self.assertEqual(hp.batch_size, 64)

    def test_yaml_overrides_only_named_fields(self):
        hp = config.load_hyperparameters(write_yaml("lr: 0.0006\nbatch_size: 32\n"))
        self.assertEqual(hp.lr, 0.0006)
        self.assertEqual(hp.batch_size, 32)
        self.assertEqual(hp.n_layer, 6)  # untouched -> default

    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError):
            config.load_hyperparameters(write_yaml("not_a_real_param: 1\n"))


if __name__ == "__main__":
    unittest.main()
