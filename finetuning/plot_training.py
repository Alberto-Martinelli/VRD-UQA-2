import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_log(path: Path) -> tuple[list, list, list, list]:
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        step = entry["current_steps"]
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_loss.append(entry["eval_loss"])
        elif "loss" in entry:
            train_steps.append(step)
            train_loss.append(entry["loss"])
    return train_steps, train_loss, eval_steps, eval_loss


def plot(log_path: Path, out_path: Path) -> None:
    train_steps, train_loss, eval_steps, eval_loss = load_log(log_path)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_steps, train_loss, label="train loss")
    if eval_loss:
        ax.plot(eval_steps, eval_loss, label="eval loss", marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(log_path.parent.name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="Path to trainer_log.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    args = parser.parse_args()

    out = args.out or args.log.parent / "loss_plot.png"
    plot(args.log, out)


if __name__ == "__main__":
    main()
