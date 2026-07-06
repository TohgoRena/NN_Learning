"""
R12 experiment: next-bit prediction for FSR-like generators.

The script compares three bit-sequence sources:
  1. LFSR
  2. The same LFSR with a nonlinear output filter
  3. Independent bits generated with secrets.randbits(1)

For each generator, it creates 200 sequences of length 19, splits them by
sequence into 120/40/40 train/validation/test sequences, converts each
sequence into sliding windows, trains a small Keras MLP, and reports metrics.
"""

from __future__ import annotations

import argparse
import csv
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import random
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


GENERATOR_NAMES = ("lfsr", "filtered_lfsr", "secrets")
LFSR_TAPS = {
    4: (0, 1),  # x^4 + x + 1
    8: (0, 4, 5, 6),  # x^8 + x^6 + x^5 + x^4 + 1
}


@dataclass(frozen=True)
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def nonzero_state(rng: np.random.Generator, n_bits: int) -> list[int]:
    """Return a random n-bit state except the all-zero state."""
    value = int(rng.integers(1, 2**n_bits))
    return [(value >> shift) & 1 for shift in reversed(range(n_bits))]


def lfsr_next_bit(state: list[int]) -> int:
    """Return the next bit for the supported primitive LFSR polynomials."""
    try:
        taps = LFSR_TAPS[len(state)]
    except KeyError as exc:
        supported = ", ".join(str(bits) for bits in sorted(LFSR_TAPS))
        raise ValueError(f"Unsupported LFSR size: {len(state)}. Supported sizes: {supported}") from exc
    next_bit = 0
    for tap in taps:
        next_bit ^= state[tap]
    return next_bit


def lfsr_sequence(length: int, rng: np.random.Generator, state_bits: int) -> list[int]:
    """Generate bits from an LFSR recurrence."""
    bits = nonzero_state(rng, state_bits)
    while len(bits) < length:
        state = bits[-state_bits:]
        bits.append(lfsr_next_bit(state))
    return bits[:length]


def nonlinear_filter(state: list[int]) -> int:
    """Small nonlinear output filter for 4-bit and 8-bit LFSR states."""
    if len(state) == 4:
        return state[0] ^ (state[1] & state[2]) ^ state[3]
    if len(state) == 8:
        return (
            state[0]
            ^ state[7]
            ^ (state[1] & state[2])
            ^ (state[3] & state[5])
            ^ (state[4] & state[6])
        )
    raise ValueError(f"Unsupported filter size: {len(state)}")


def filtered_lfsr_sequence(length: int, rng: np.random.Generator, state_bits: int) -> list[int]:
    """Generate bits from an LFSR state passed through a nonlinear filter."""
    state = nonzero_state(rng, state_bits)
    bits: list[int] = []
    for _ in range(length):
        bits.append(nonlinear_filter(state))
        state = state[1:] + [lfsr_next_bit(state)]
    return bits


def secrets_sequence(
    length: int,
    rng: np.random.Generator | None = None,
    state_bits: int | None = None,
) -> list[int]:
    """Generate independent bits with Python's cryptographic RNG."""
    del rng, state_bits
    return [secrets.randbits(1) for _ in range(length)]


def make_sequences(
    generator: Callable[[int, np.random.Generator, int], list[int]],
    count: int,
    length: int,
    rng: np.random.Generator,
    state_bits: int,
) -> np.ndarray:
    return np.array([generator(length, rng, state_bits) for _ in range(count)], dtype=np.float32)


def windows_from_sequences(sequences: np.ndarray, input_window: int) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for seq in sequences:
        for start in range(0, len(seq) - input_window):
            xs.append(seq[start : start + input_window])
            ys.append(seq[start + input_window])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def build_dataset(
    generator_name: str,
    seed: int,
    sequence_count: int,
    sequence_length: int,
    input_window: int,
    state_bits: int,
    split: tuple[int, int, int],
) -> Dataset:
    generators: dict[str, Callable[[int, np.random.Generator, int], list[int]]] = {
        "lfsr": lfsr_sequence,
        "filtered_lfsr": filtered_lfsr_sequence,
        "secrets": secrets_sequence,
    }
    if generator_name not in generators:
        raise ValueError(f"Unknown generator: {generator_name}")

    train_count, val_count, test_count = split
    if train_count + val_count + test_count != sequence_count:
        raise ValueError("Split counts must add up to sequence_count.")

    rng = np.random.default_rng(seed)
    sequences = make_sequences(generators[generator_name], sequence_count, sequence_length, rng, state_bits)
    train_seq = sequences[:train_count]
    val_seq = sequences[train_count : train_count + val_count]
    test_seq = sequences[train_count + val_count :]

    x_train, y_train = windows_from_sequences(train_seq, input_window)
    x_val, y_val = windows_from_sequences(val_seq, input_window)
    x_test, y_test = windows_from_sequences(test_seq, input_window)

    return Dataset(x_train, y_train, x_val, y_val, x_test, y_test)


def make_model(input_window: int, hidden_units: tuple[int, ...], learning_rate: float) -> tf.keras.Model:
    layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=(input_window,))]
    for units in hidden_units:
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def parameter_count(input_window: int, hidden_units: tuple[int, ...]) -> int:
    layer_sizes = (input_window, *hidden_units, 1)
    return sum((layer_sizes[i] * layer_sizes[i + 1]) + layer_sizes[i + 1] for i in range(len(layer_sizes) - 1))


def majority_baseline(y_train: np.ndarray, y_true: np.ndarray) -> dict[str, float | int]:
    majority = int(np.mean(y_train) >= 0.5)
    y_pred = np.full_like(y_true, fill_value=majority)
    return metric_dict(y_true, y_pred, prefix="baseline_")


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}tn": int(tn),
        f"{prefix}fp": int(fp),
        f"{prefix}fn": int(fn),
        f"{prefix}tp": int(tp),
    }


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def train_once(
    generator_name: str,
    run_index: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, float | int | str], tf.keras.callbacks.History]:
    set_all_seeds(seed)
    dataset = build_dataset(
        generator_name=generator_name,
        seed=seed,
        sequence_count=args.sequence_count,
        sequence_length=args.sequence_length,
        input_window=args.input_window,
        state_bits=args.state_bits,
        split=(args.train_sequences, args.val_sequences, args.test_sequences),
    )

    model = make_model(args.input_window, args.hidden_units, args.learning_rate)
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        )
    ]

    start = time.perf_counter()
    history = model.fit(
        dataset.x_train,
        dataset.y_train,
        validation_data=(dataset.x_val, dataset.y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - start

    probabilities = model.predict(dataset.x_test, verbose=0).reshape(-1)
    y_pred = (probabilities >= 0.5).astype(np.int32)
    y_true = dataset.y_test.astype(np.int32)

    train_eval = model.evaluate(dataset.x_train, dataset.y_train, verbose=0)
    val_eval = model.evaluate(dataset.x_val, dataset.y_val, verbose=0)
    test_eval = model.evaluate(dataset.x_test, dataset.y_test, verbose=0)

    row: dict[str, float | int | str] = {
        "generator": generator_name,
        "run": run_index,
        "seed": seed,
        "state_bits": args.state_bits,
        "input_window": args.input_window,
        "hidden_units": "-".join(str(unit) for unit in args.hidden_units),
        "parameters": parameter_count(args.input_window, args.hidden_units),
        "train_samples": len(dataset.y_train),
        "val_samples": len(dataset.y_val),
        "test_samples": len(dataset.y_test),
        "epochs_ran": len(history.history["loss"]),
        "train_loss": float(train_eval[0]),
        "train_accuracy": float(train_eval[1]),
        "val_loss": float(val_eval[0]),
        "val_accuracy": float(val_eval[1]),
        "test_loss": float(test_eval[0]),
        "test_accuracy_from_keras": float(test_eval[1]),
        "elapsed_seconds": elapsed,
    }
    row.update(metric_dict(y_true, y_pred))
    row.update(majority_baseline(dataset.y_train.astype(np.int32), y_true))
    return row, history


def write_results_csv(rows: Iterable[dict[str, float | int | str]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(rows: list[dict[str, float | int | str]], path: Path) -> list[dict[str, float | str]]:
    numeric_keys = [
        "train_accuracy",
        "val_accuracy",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "test_loss",
        "baseline_accuracy",
        "baseline_balanced_accuracy",
        "elapsed_seconds",
        "epochs_ran",
    ]
    summary: list[dict[str, float | str]] = []
    for generator in GENERATOR_NAMES:
        group = [row for row in rows if row["generator"] == generator]
        if not group:
            continue
        item: dict[str, float | str] = {"generator": generator}
        for key in numeric_keys:
            values = np.array([float(row[key]) for row in group], dtype=np.float64)
            item[f"{key}_mean"] = float(values.mean())
            item[f"{key}_std"] = float(values.std(ddof=0))
        summary.append(item)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def plot_summary(summary: list[dict[str, float | str]], out_dir: Path) -> None:
    labels = [str(row["generator"]) for row in summary]
    x = np.arange(len(labels))
    width = 0.35

    accuracy = np.array([float(row["accuracy_mean"]) for row in summary])
    accuracy_std = np.array([float(row["accuracy_std"]) for row in summary])
    baseline = np.array([float(row["baseline_accuracy_mean"]) for row in summary])
    baseline_std = np.array([float(row["baseline_accuracy_std"]) for row in summary])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, accuracy, width, yerr=accuracy_std, label="MLP")
    ax.bar(x + width / 2, baseline, width, yerr=baseline_std, label="majority baseline")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("test accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "test_accuracy_summary.png", dpi=160)
    plt.close(fig)

    balanced = np.array([float(row["balanced_accuracy_mean"]) for row in summary])
    balanced_std = np.array([float(row["balanced_accuracy_std"]) for row in summary])
    baseline_balanced = np.array([float(row["baseline_balanced_accuracy_mean"]) for row in summary])
    baseline_balanced_std = np.array([float(row["baseline_balanced_accuracy_std"]) for row in summary])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, balanced, width, yerr=balanced_std, label="MLP")
    ax.bar(
        x + width / 2,
        baseline_balanced,
        width,
        yerr=baseline_balanced_std,
        label="majority baseline",
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("test balanced accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "balanced_accuracy_summary.png", dpi=160)
    plt.close(fig)


def plot_learning_curves(
    histories: dict[str, list[tf.keras.callbacks.History]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for generator, runs in histories.items():
        max_len = max(len(history.history["loss"]) for history in runs)
        padded_loss = np.full((len(runs), max_len), np.nan)
        padded_val_loss = np.full((len(runs), max_len), np.nan)
        for idx, history in enumerate(runs):
            loss = np.array(history.history["loss"], dtype=np.float64)
            val_loss = np.array(history.history["val_loss"], dtype=np.float64)
            padded_loss[idx, : len(loss)] = loss
            padded_val_loss[idx, : len(val_loss)] = val_loss
        axes[0].plot(np.nanmean(padded_loss, axis=0), label=generator)
        axes[1].plot(np.nanmean(padded_val_loss, axis=0), label=generator)

    axes[0].set_title("train loss")
    axes[1].set_title("validation loss")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylabel("binary crossentropy")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=160)
    plt.close(fig)


def parse_hidden_units(value: str, state_bits: int) -> tuple[int, ...]:
    if value == "auto":
        return (8, 4) if state_bits == 4 else (64, 32, 16)
    units = tuple(int(part) for part in value.split(",") if part.strip())
    if not units or any(unit <= 0 for unit in units):
        raise ValueError("hidden_units must be 'auto' or a comma-separated list of positive integers.")
    return units


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the R12 next-bit prediction experiment.")
    parser.add_argument("--output-dir", type=Path, default=Path("r12_results"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--state-bits", type=int, default=4, choices=sorted(LFSR_TAPS))
    parser.add_argument("--sequence-count", type=int, default=200)
    parser.add_argument("--sequence-length", type=int, default=19)
    parser.add_argument("--input-window", type=int, default=4)
    parser.add_argument("--hidden-units", type=str, default="auto")
    parser.add_argument("--train-sequences", type=int, default=120)
    parser.add_argument("--val-sequences", type=int, default=40)
    parser.add_argument("--test-sequences", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])
    args = parser.parse_args()
    args.hidden_units = parse_hidden_units(args.hidden_units, args.state_bits)
    return args


def main() -> None:
    args = parse_args()
    if args.input_window >= args.sequence_length:
        raise ValueError("input_window must be smaller than sequence_length.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    histories: dict[str, list[tf.keras.callbacks.History]] = {name: [] for name in GENERATOR_NAMES}

    print(f"State bits: {args.state_bits}")
    print(f"Input window: {args.input_window} bits")
    print(f"Hidden units: {args.hidden_units}")
    print(f"Model parameters: {parameter_count(args.input_window, args.hidden_units)}")
    print(f"Output directory: {args.output_dir.resolve()}")

    for generator_index, generator_name in enumerate(GENERATOR_NAMES):
        print(f"\n=== {generator_name} ===")
        for run_index in range(1, args.runs + 1):
            seed = args.base_seed + generator_index * 1000 + run_index
            row, history = train_once(generator_name, run_index, seed, args)
            rows.append(row)
            histories[generator_name].append(history)
            print(
                "run {run}: test_acc={acc:.3f}, bal_acc={bal:.3f}, "
                "baseline={base:.3f}, epochs={epochs}, time={sec:.1f}s".format(
                    run=run_index,
                    acc=float(row["accuracy"]),
                    bal=float(row["balanced_accuracy"]),
                    base=float(row["baseline_accuracy"]),
                    epochs=int(row["epochs_ran"]),
                    sec=float(row["elapsed_seconds"]),
                )
            )

    write_results_csv(rows, args.output_dir / "results.csv")
    summary = write_summary_csv(rows, args.output_dir / "summary.csv")
    plot_summary(summary, args.output_dir)
    plot_learning_curves(histories, args.output_dir)

    print("\nSummary")
    for row in summary:
        print(
            "{generator}: acc={acc:.3f}+/-{acc_std:.3f}, "
            "bal_acc={bal:.3f}+/-{bal_std:.3f}, "
            "baseline={base:.3f}+/-{base_std:.3f}".format(
                generator=row["generator"],
                acc=float(row["accuracy_mean"]),
                acc_std=float(row["accuracy_std"]),
                bal=float(row["balanced_accuracy_mean"]),
                bal_std=float(row["balanced_accuracy_std"]),
                base=float(row["baseline_accuracy_mean"]),
                base_std=float(row["baseline_accuracy_std"]),
            )
        )
    print("\nSaved: results.csv, summary.csv, test_accuracy_summary.png, balanced_accuracy_summary.png, learning_curves.png")


if __name__ == "__main__":
    main()
