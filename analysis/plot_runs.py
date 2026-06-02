import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Explicit map of each canonical experiment to its representative log file, in
# chronological order. Curated by config signature (lr/bs/dropout/vocab/n_layer
# + param count) so labels are exact; duplicate/confirmation runs are omitted.
# (label, log filename, status)
EXPERIMENTS = [
    ("Baseline (SGD)",      "baseline.log",                     "baseline"),
    ("AdamW lr 3e-4",       "mainrun_2026-06-02T13-57-57.log",  "win"),
    ("+ warmup 6e-4",       "mainrun_2026-06-02T14-07-18.log",  "win"),
    ("+ residual init",     "mainrun_2026-06-02T14-16-14.log",  "win"),
    ("+ LR 1e-3",           "mainrun_2026-06-02T14-18-55.log",  "win"),
    ("LR 1.5e-3",           "mainrun_2026-06-02T14-28-31.log",  "reject"),
    ("batch 32",            "mainrun_2026-06-02T14-31-50.log",  "reject"),
    ("depth 8",             "mainrun_2026-06-02T14-57-58.log",  "reject"),
    ("+ RoPE",              "mainrun_2026-06-02T15-04-19.log",  "win"),
    ("SwiGLU",              "mainrun_2026-06-02T15-20-38.log",  "reject"),
    ("dropout 0",           "mainrun_2026-06-02T15-26-27.log",  "reject"),
    ("vocab 8000",          "mainrun_2026-06-02T15-37-14.log",  "reject"),
    ("+ per-title mask",    "mainrun.log",                      "win"),
]

# Val-loss lineage for the curve overlay (the accepted chain only).
LINEAGE = ["Baseline (SGD)", "AdamW lr 3e-4", "+ warmup 6e-4", "+ residual init",
           "+ LR 1e-3", "+ RoPE", "+ per-title mask"]

STATUS_COLOR = {"baseline": "#444444", "win": "#2ca02c", "reject": "#d62728"}


def parse_log(path: Path) -> dict | None:
    """Parse one JSON-lines log into config, params, and train/val curves.

    Returns None if the log has no validation steps.
    """
    cfg, params, t0 = {}, None, None
    train_steps, train_loss, val_steps, val_loss = [], [], [], []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = e.get("event")
        if t0 is None:
            t0 = e.get("timestamp")
        if ev == "hyperparameters_configured":
            cfg = e
        elif ev == "model_info":
            params = e.get("parameters_count")
        elif ev == "training_step":
            train_steps.append(e["step"])
            train_loss.append(e["loss"])
        elif ev == "validation_step":
            val_steps.append(e["step"])
            val_loss.append(e["loss"])
    if not val_loss:
        return None
    return {
        "path": path, "cfg": cfg, "params": params, "t0": t0,
        "train_steps": train_steps, "train_loss": train_loss,
        "val_steps": val_steps, "val_loss": val_loss,
        "final_val": val_loss[-1],
    }


def load_runs(logs_dir: Path) -> list[dict]:
    """Load the canonical runs named in EXPERIMENTS, in order, labelled and tagged."""
    runs = []
    for label, fname, status in EXPERIMENTS:
        r = parse_log(logs_dir / fname)
        if r is None:
            print(f"warning: missing/empty log for {label}: {fname}")
            continue
        r["label"], r["status"] = label, status
        runs.append(r)
    return runs


def plot_journey(runs: list[dict], out: Path) -> None:
    """Plot final val loss per experiment with a running-best step line."""
    labels = [r["label"] for r in runs]
    finals = [r["final_val"] for r in runs]
    colors = [STATUS_COLOR[r["status"]] for r in runs]

    running_best, best = [], 1e9
    for f, r in zip(finals, runs):
        if r["status"] in ("win", "baseline"):
            best = min(best, f)
        running_best.append(best)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(runs))
    ax.scatter(x, finals, c=colors, s=90, zorder=3)
    ax.step(x, running_best, where="post", color="#1f77b4", lw=2, label="running best", zorder=2)
    for xi, f in zip(x, finals):
        ax.annotate(f"{f:.4f}", (xi, f), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("final validation loss")
    ax.set_title("Mainrun: experimentation journey (1.754 -> 1.2046)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=lbl)
               for lbl, c in [("win", STATUS_COLOR["win"]), ("rejected", STATUS_COLOR["reject"])]]
    handles.append(plt.Line2D([], [], color="#1f77b4", lw=2, label="running best"))
    ax.legend(handles=handles)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "journey.png", dpi=150)
    print(f"wrote {out / 'journey.png'}")


def plot_curves(runs: list[dict], out: Path) -> None:
    """Plot baseline-vs-best train loss and the winning-lineage val loss curves."""
    by_label = {r["label"]: r for r in runs}
    fig, (axt, axv) = plt.subplots(1, 2, figsize=(13, 5))

    baseline = by_label.get("Baseline (SGD)")
    best = by_label.get("+ per-title mask") or by_label.get("+ RoPE")

    for r, name, color in [(baseline, "baseline (SGD)", "#d62728"), (best, "best", "#2ca02c")]:
        if r:
            axt.plot(r["train_steps"], r["train_loss"], color=color, lw=1, alpha=0.8, label=name)
    axt.set_title("train loss (per-token CE)")
    axt.set_xlabel("step")
    axt.set_ylabel("loss")
    axt.legend()
    axt.grid(True, alpha=0.3)

    cmap = plt.get_cmap("viridis")
    chosen = [lbl for lbl in LINEAGE if lbl in by_label]
    for i, lbl in enumerate(chosen):
        r = by_label[lbl]
        axv.plot(r["val_steps"], r["val_loss"], lw=1.5,
                 color=cmap(i / max(1, len(chosen) - 1)), label=lbl)
    axv.set_title("validation loss (winning lineage)")
    axv.set_xlabel("step")
    axv.set_ylabel("loss")
    axv.legend(fontsize=8)
    axv.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=150)
    print(f"wrote {out / 'curves.png'}")


def main() -> None:
    """Parse logs from --logs and write the journey and curves figures to --out."""
    ap = argparse.ArgumentParser(description="Plot the Mainrun experimentation journey from logs.")
    ap.add_argument("--logs", default="mainrun/logs", help="directory of JSON-lines run logs")
    ap.add_argument("--out", default="analysis/figures", help="directory to write figures to")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(Path(args.logs))
    print(f"{'file':40} {'params':>11} {'final_val':>10}  label")
    for r in runs:
        print(f"{r['path'].name:40} {str(r['params']):>11} {r['final_val']:>10.6f}  {r['label']}")

    plot_journey(runs, out)
    plot_curves(runs, out)


if __name__ == "__main__":
    main()
