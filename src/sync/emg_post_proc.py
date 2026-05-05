#!/usr/bin/env python3

"""
Process EMG parquet for ML/analysis.

Pipeline:
1. Load raw EMG parquet
2. Apply 20-450 Hz bandpass filter per epoch
3. Apply centered RMS window
4. Apply per-epoch baseline correction using -5 to 0 s
5. Align all epochs to a common native-rate time grid
6. Export processed parquet

Expected columns:
- time_sec
- run_id
- task_instance
- EMG columns containing "EMG" and ending with "(mV)"

Example:
python process_emg_rms_baseline_aligned.py \
  ./data/processed/combined/data_packet/emg_full.parquet \
  ./data/processed/combined/data_packet/emg_rms_baseline_aligned.parquet
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import polars as pl
from scipy.signal import butter, sosfiltfilt


def get_emg_cols(df: pl.DataFrame) -> list[str]:
    """Find EMG signal columns."""
    return [
        c for c in df.columns
        if "EMG" in c and c.endswith("(mV)")
    ]


def clean_signal(series: pl.Series) -> np.ndarray:
    """
    Convert a Polars Series to a clean numpy array for filtering.

    Handles NaN/null values by interpolation and edge filling.
    """
    return (
        series
        .fill_nan(None)
        .interpolate()
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
        .to_numpy()
    )


def estimate_fs(epoch_df: pl.DataFrame, time_col: str = "time_sec") -> float:
    """Estimate sampling frequency from median timestamp difference."""
    time = epoch_df[time_col].to_numpy()

    if len(time) < 3:
        raise ValueError("Not enough samples to estimate sampling rate.")

    dt = np.diff(time)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]

    if len(dt) == 0:
        raise ValueError(
            "Could not estimate sampling rate: no positive timestamp differences."
        )

    return float(1.0 / np.median(dt))


def filter_rms_epoch(
    epoch_df: pl.DataFrame,
    emg_cols: list[str],
    lowcut: float = 20.0,
    highcut: float = 450.0,
    rms_window_sec: float = 0.100,
    time_col: str = "time_sec",
) -> pl.DataFrame:
    """
    Apply bandpass filter and RMS envelope to one epoch.

    Important:
    This should only be called on one continuous epoch at a time.
    """
    epoch_df = epoch_df.sort(time_col)

    fs = estimate_fs(epoch_df, time_col=time_col)
    nyquist = fs / 2.0

    if highcut >= nyquist:
        raise ValueError(
            f"Highcut {highcut:.2f} Hz is >= Nyquist {nyquist:.2f} Hz. "
            f"Estimated sampling rate is {fs:.2f} Hz."
        )

    sos = butter(
        N=2,
        Wn=[lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    bandpassed = epoch_df.with_columns(
        [
            pl.Series(
                name=col,
                values=sosfiltfilt(
                    sos,
                    clean_signal(epoch_df[col]),
                ),
            )
            for col in emg_cols
        ]
    )

    rms_window_samples = int(round(rms_window_sec * fs))
    rms_window_samples = max(rms_window_samples, 1)

    rms = bandpassed.with_columns(
        [
            (
                pl.col(col)
                .pow(2)
                .rolling_mean(
                    window_size=rms_window_samples,
                    center=True,
                )
                .sqrt()
                .alias(col)
            )
            for col in emg_cols
        ]
    )

    return rms


def filter_rms_per_epoch(
    emg_df: pl.DataFrame,
    emg_cols: list[str],
    group_cols: list[str],
    lowcut: float = 20.0,
    highcut: float = 450.0,
    rms_window_sec: float = 0.100,
    time_col: str = "time_sec",
) -> pl.DataFrame:
    """
    Apply bandpass + RMS independently to each epoch.

    This prevents filtering across task/run boundaries where time_sec resets.
    """
    processed_epochs: list[pl.DataFrame] = []

    for key, epoch in emg_df.group_by(group_cols, maintain_order=True):
        epoch = epoch.sort(time_col)

        if epoch.height < 50:
            print(f"Warning: skipping filtering for tiny epoch {key}, rows={epoch.height}")
            processed_epochs.append(epoch)
            continue

        processed = filter_rms_epoch(
            epoch,
            emg_cols=emg_cols,
            lowcut=lowcut,
            highcut=highcut,
            rms_window_sec=rms_window_sec,
            time_col=time_col,
        )

        processed_epochs.append(processed)

    if not processed_epochs:
        raise ValueError("No epochs were processed.")

    return (
        pl.concat(processed_epochs, how="vertical")
        .sort(group_cols + [time_col])
    )


def apply_baseline_correction(
    df: pl.DataFrame,
    emg_cols: list[str],
    group_cols: list[str],
    time_col: str = "time_sec",
    baseline_start: float = -5.0,
    baseline_end: float = 0.0,
) -> pl.DataFrame:
    """
    Subtract per-epoch baseline from each EMG channel.

    Baseline is the mean RMS during:
    baseline_start <= time_sec < baseline_end
    """
    baseline_df = (
        df
        .filter(
            (pl.col(time_col) >= baseline_start)
            & (pl.col(time_col) < baseline_end)
        )
        .group_by(group_cols)
        .agg(
            [
                pl.col(c)
                .fill_nan(None)
                .drop_nulls()
                .mean()
                .alias(f"{c}__baseline")
                for c in emg_cols
            ]
        )
    )

    out = df.join(baseline_df, on=group_cols, how="left")

    out = out.with_columns(
        [
            (
                pl.col(c).fill_nan(None)
                - pl.col(f"{c}__baseline")
            ).alias(c)
            for c in emg_cols
        ]
    )

    out = out.drop([f"{c}__baseline" for c in emg_cols])

    return out


def align_emg_epochs_to_common_time(
    emg_df: pl.DataFrame,
    emg_cols: list[str],
    group_cols: list[str],
    time_col: str = "time_sec",
    time_min: float = -5.0,
    time_max: float = 15.0,
) -> pl.DataFrame:
    """
    Align processed EMG epochs to a common native-rate time grid.

    This fixes tiny floating-point / fractional-sample timestamp offsets
    between epochs without downsampling.

    The exported dataframe will have:
    - time_sec: aligned common time grid
    - time_sec_raw_start: original first timestamp for that epoch
    - time_sec_raw_end: original final timestamp for that epoch
    - sample_idx: sample index on the aligned grid
    """

    emg_df = emg_df.sort(group_cols + [time_col])

    dt = (
        emg_df
        .with_columns(
            pl.col(time_col)
            .diff()
            .over(group_cols)
            .alias("_dt")
        )
        .select(pl.col("_dt").drop_nulls().median())
        .item()
    )

    if dt is None or not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid median dt: {dt}")

    fs = 1.0 / dt

    epoch_limits = (
        emg_df
        .group_by(group_cols)
        .agg(
            [
                pl.col(time_col).min().alias("t_min"),
                pl.col(time_col).max().alias("t_max"),
            ]
        )
    )

    # Use only the time span that all epochs actually cover.
    actual_time_min = max(time_min, epoch_limits["t_min"].max())
    actual_time_max = min(time_max, epoch_limits["t_max"].min())

    if actual_time_max <= actual_time_min:
        raise ValueError(
            f"Invalid common time range: {actual_time_min} to {actual_time_max}"
        )

    t_grid = np.arange(actual_time_min, actual_time_max, dt)
    sample_idx = np.arange(len(t_grid), dtype=np.int64)

    print(f"Estimated native fs: {fs:.3f} Hz")
    print(f"Median dt: {dt:.9f} s")
    print(f"Aligned time range: {actual_time_min:.6f} to {actual_time_max:.6f} s")
    print(f"Aligned samples per epoch: {len(t_grid):,}")

    aligned_epochs: list[pl.DataFrame] = []

    meta_cols = [
        c for c in emg_df.columns
        if c not in emg_cols and c != time_col
    ]

    for key, g in emg_df.group_by(group_cols, maintain_order=True):
        g = g.sort(time_col)

        if not isinstance(key, tuple):
            key = (key,)

        t_raw = g[time_col].to_numpy()

        rows = {
            time_col: t_grid,
            f"{time_col}_raw_start": float(g[time_col][0]),
            f"{time_col}_raw_end": float(g[time_col][-1]),
            "sample_idx": sample_idx,
        }

        # Preserve group columns.
        for col_name, value in zip(group_cols, key):
            rows[col_name] = value

        # Preserve metadata as first value in each epoch.
        for c in meta_cols:
            if c not in group_cols:
                rows[c] = g[c][0]

        # Interpolate EMG channels onto common grid.
        for c in emg_cols:
            y = g[c].to_numpy()
            valid = np.isfinite(t_raw) & np.isfinite(y)

            if valid.sum() < 2:
                rows[c] = np.full(len(t_grid), np.nan)
            else:
                rows[c] = np.interp(
                    t_grid,
                    t_raw[valid],
                    y[valid],
                    left=np.nan,
                    right=np.nan,
                )

        aligned_epochs.append(pl.DataFrame(rows))

    out = pl.concat(aligned_epochs, how="vertical")

    # Critical:
    # np.interp can create NaNs at edges.
    # Convert NaNs to nulls so Polars means ignore them correctly.
    out = out.with_columns(
        [
            pl.when(pl.col(c).is_nan())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in emg_cols
        ]
    )

    return out.sort(group_cols + [time_col])


def add_processing_metadata(
    df: pl.DataFrame,
    lowcut: float,
    highcut: float,
    rms_window_sec: float,
    baseline_start: float,
    baseline_end: float,
    aligned: bool,
) -> pl.DataFrame:
    """Add simple metadata columns documenting the processing."""
    return df.with_columns(
        [
            pl.lit(lowcut).alias("emg_filter_lowcut_hz"),
            pl.lit(highcut).alias("emg_filter_highcut_hz"),
            pl.lit(rms_window_sec).alias("emg_rms_window_sec"),
            pl.lit(baseline_start).alias("emg_baseline_start_sec"),
            pl.lit(baseline_end).alias("emg_baseline_end_sec"),
            pl.lit(True).alias("emg_bandpass_applied"),
            pl.lit(True).alias("emg_rms_applied"),
            pl.lit(True).alias("emg_baseline_corrected"),
            pl.lit(aligned).alias("emg_time_aligned"),
        ]
    )


def process_emg_file(
    input_path: str | Path,
    output_path: str | Path,
    lowcut: float = 20.0,
    highcut: float = 450.0,
    rms_window_sec: float = 0.100,
    baseline_start: float = -5.0,
    baseline_end: float = 0.0,
    align_time: bool = True,
    align_time_min: float = -5.0,
    align_time_max: float = 15.0,
    group_cols: list[str] | None = None,
    time_col: str = "time_sec",
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if group_cols is None:
        group_cols = ["run_id", "task_instance"]

    print(f"Loading: {input_path}")
    emg_df = pl.read_parquet(input_path)

    required_cols = group_cols + [time_col]
    missing_cols = [c for c in required_cols if c not in emg_df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    emg_cols = get_emg_cols(emg_df)

    if not emg_cols:
        raise ValueError(
            'No EMG columns found. Expected columns containing "EMG" '
            'and ending with "(mV)".'
        )

    print(f"Input rows: {emg_df.height:,}")
    print(f"Input columns: {emg_df.width}")
    print(f"EMG channels: {len(emg_cols)}")
    print(f"Epoch group columns: {group_cols}")

    print("Applying bandpass + RMS per epoch...")
    processed = filter_rms_per_epoch(
        emg_df,
        emg_cols=emg_cols,
        group_cols=group_cols,
        lowcut=lowcut,
        highcut=highcut,
        rms_window_sec=rms_window_sec,
        time_col=time_col,
    )

    print("Applying baseline correction...")
    processed = apply_baseline_correction(
        processed,
        emg_cols=emg_cols,
        group_cols=group_cols,
        time_col=time_col,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
    )

    if align_time:
        print("Aligning epochs to common native-rate time grid...")
        processed = align_emg_epochs_to_common_time(
            processed,
            emg_cols=emg_cols,
            group_cols=group_cols,
            time_col=time_col,
            time_min=align_time_min,
            time_max=align_time_max,
        )
    else:
        processed = processed.with_columns(
            [
                pl.when(pl.col(c).is_nan())
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in emg_cols
            ]
        )

    processed = add_processing_metadata(
        processed,
        lowcut=lowcut,
        highcut=highcut,
        rms_window_sec=rms_window_sec,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        aligned=align_time,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing: {output_path}")
    processed.write_parquet(output_path)

    print("Done.")
    print(f"Output rows: {processed.height:,}")
    print(f"Output columns: {processed.width}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bandpass filter, RMS, baseline-correct, and optionally align "
            "EMG parquet data."
        )
    )

    parser.add_argument(
        "input",
        help="Input raw EMG parquet file.",
    )

    parser.add_argument(
        "output",
        help="Output processed EMG parquet file.",
    )

    parser.add_argument(
        "--lowcut",
        type=float,
        default=20.0,
        help="Bandpass low cutoff in Hz. Default: 20.",
    )

    parser.add_argument(
        "--highcut",
        type=float,
        default=450.0,
        help="Bandpass high cutoff in Hz. Default: 450.",
    )

    parser.add_argument(
        "--rms-window",
        type=float,
        default=0.100,
        help="Centered RMS window in seconds. Default: 0.100.",
    )

    parser.add_argument(
        "--baseline-start",
        type=float,
        default=-5.0,
        help="Baseline window start in seconds. Default: -5.",
    )

    parser.add_argument(
        "--baseline-end",
        type=float,
        default=0.0,
        help="Baseline window end in seconds. Default: 0.",
    )

    parser.add_argument(
        "--align-time-min",
        type=float,
        default=-5.0,
        help="Minimum aligned time. Default: -5.",
    )

    parser.add_argument(
        "--align-time-max",
        type=float,
        default=15.0,
        help="Maximum aligned time. Default: 15.",
    )

    parser.add_argument(
        "--no-align-time",
        action="store_true",
        help="Disable timestamp alignment.",
    )

    parser.add_argument(
        "--time-col",
        default="time_sec",
        help="Time column name. Default: time_sec.",
    )

    parser.add_argument(
        "--group-cols",
        nargs="+",
        default=["run_id", "task_instance"],
        help="Columns defining one EMG epoch. Default: run_id task_instance.",
    )

    args = parser.parse_args()

    process_emg_file(
        input_path=args.input,
        output_path=args.output,
        lowcut=args.lowcut,
        highcut=args.highcut,
        rms_window_sec=args.rms_window,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
        align_time=not args.no_align_time,
        align_time_min=args.align_time_min,
        align_time_max=args.align_time_max,
        group_cols=args.group_cols,
        time_col=args.time_col,
    )


if __name__ == "__main__":
    main()