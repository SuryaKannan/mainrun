import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from plot_runs import parse_log

REPO = Path(__file__).resolve().parent.parent
MAINRUN = REPO / "mainrun"


def run_one(name: str, overrides: dict) -> float:
    """Run train.py once with the given overrides; return its final validation loss."""
    overrides = {**overrides, "log_file": f"./logs/sweep_{name}.log"}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(overrides, f)
        config_path = f.name
    subprocess.run([sys.executable, "train.py", "--config", config_path], cwd=MAINRUN, check=True)
    run = parse_log(MAINRUN / "logs" / f"sweep_{name}.log")
    return run["final_val"] if run else float("nan")


def main() -> None:
    """Run every variation in a YAML sweep spec and print results sorted by val loss."""
    ap = argparse.ArgumentParser(description="Run a hyperparameter sweep from a YAML spec.")
    ap.add_argument("spec", help="YAML file with an optional 'base' map and a list of 'runs'")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.spec).read_text())
    base = spec.get("base", {})

    results = []
    for run in spec["runs"]:
        name = run.pop("name")
        overrides = {**base, **run}
        print(f"=== {name}: {overrides} ===")
        final = run_one(name, overrides)
        results.append((name, final))
        print(f"  -> final val loss: {final:.6f}")

    print("\nsweep results (best first):")
    for name, final in sorted(results, key=lambda r: r[1]):
        print(f"  {final:.6f}  {name}")


if __name__ == "__main__":
    main()
