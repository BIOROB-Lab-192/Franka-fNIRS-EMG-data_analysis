"""
data_combiner.py — Multi-Stream Data Combiner
==============================================
Combines EMG, Robot, and fNIRS data into:
1. A 100 Hz resampled combined file (easy sharing)
2. A full-resolution data packet (detailed analysis)

The 100 Hz file uses a -5 to 15s time grid per epoch (2001 points).
fNIRS covers the full range including the 5s pre-task baseline.
EMG and robot are null during -5 to 0s (no data in that window).

Franka_ee/q/dq/tau_J columns are string-encoded arrays — carried as-is
through the nearest-neighbor resampling, not interpolated.

Usage:
    python -m src.sync.data_combiner
    python -m src.sync.data_combiner --emg PATH --robot PATH --fnirs PATH --output DIR

    # or import:
    from src.sync.data_combiner import build_combined_export
    build_combined_export(emg_path, robot_path, fnirs_path, output_dir)
"""

import polars as pl
from pathlib import Path


# ── Constants ──────────────────────────────────────────────

TARGET_HZ = 100
EPOCH_TMIN = -5.0   # fNIRS baseline starts at -5s
EPOCH_TMAX = 15.0   # Task window ends at 15s
N_POINTS = int((EPOCH_TMAX - EPOCH_TMIN) * TARGET_HZ) + 1  # 2001


# ═══════════════════════════════════════════════════════════════
# 1. LOADERS
# ═══════════════════════════════════════════════════════════════

def load_emg_epochs(path: str | Path) -> pl.DataFrame:
    """
    Load EMG processed parquet.

    Normalizes:
      'time' column → 'time_sec' (0–15s, relative to epoch start)

    Schema after loading:
      time_sec, [sensor channels...], task_instance, run_id,
      is_robot, participant, epoch_start
    """
    df = pl.read_parquet(path)

    if "time" in df.columns:
        df = df.rename({"time": "time_sec"})

    # Normalize to epoch-relative time (0 to 15s)
    if "epoch_start" in df.columns:
        df = df.with_columns(
            (pl.col("time_sec") - pl.col("epoch_start")).alias("time_sec")
        )

    return df


def load_robot_epochs(path: str | Path) -> pl.DataFrame:
    """
    Load robot processed CSV.

    Computes:
      time_sec = timestamp - sequence_start_timestamp (0–15s)

    Franka_ee / Franka_q / Franka_dq / Franka_tau_J are string-encoded
    arrays — they travel with their timestamp row and are not interpolated.
    """
    df = pl.read_csv(path)

    if "time_sec" not in df.columns and "sequence_start_timestamp" in df.columns:
        df = df.with_columns(
            (pl.col("timestamp") - pl.col("sequence_start_timestamp")).alias("time_sec")
        )

    return df


def load_fnirs_epochs(path: str | Path) -> pl.DataFrame:
    """
    Load fNIRS processed CSV.

    Already has 'time_sec' (-5 to 15s, relative to task onset).
    task_instance is 1-indexed — maps directly to EMG/robot task_instance.
    """
    return pl.read_csv(path)


# ═══════════════════════════════════════════════════════════════
# 2. EPOCH VERIFICATION
# ═══════════════════════════════════════════════════════════════

def verify_epoch_alignment(
    emg_df: pl.DataFrame,
    robot_df: pl.DataFrame,
    fnirs_df: pl.DataFrame,
) -> dict:
    """
    Check that all streams have matching epochs before combining.

    EMG and fNIRS should have identical (run_id, task_instance) sets.
    Robot should be a subset (only robot sessions have robot data).
    """
    emg_epochs = set(emg_df.select("run_id", "task_instance").unique().rows())
    fnirs_epochs = set(fnirs_df.select("run_id", "task_instance").unique().rows())
    robot_epochs = set(robot_df.select("run_id", "task_instance").unique().rows())

    common_ef = emg_epochs & fnirs_epochs
    emg_only = emg_epochs - fnirs_epochs
    fnirs_only = fnirs_epochs - emg_epochs
    robot_not_in_emg = robot_epochs - emg_epochs

    print(f"  EMG:   {len(emg_epochs)} epochs")
    print(f"  fNIRS: {len(fnirs_epochs)} epochs")
    print(f"  Robot: {len(robot_epochs)} epochs")
    print(f"  Common (EMG ∩ fNIRS): {len(common_ef)}")

    if emg_only:
        print(f"  ⚠ EMG-only epochs: {sorted(emg_only)[:5]}")
    if fnirs_only:
        print(f"  ⚠ fNIRS-only epochs: {sorted(fnirs_only)[:5]}")
    if robot_not_in_emg:
        print(f"  ⚠ Robot epochs not in EMG: {len(robot_not_in_emg)}")

    for name, df in [("EMG", emg_df), ("Robot", robot_df), ("fNIRS", fnirs_df)]:
        if df.height > 0:
            print(f"  {name} time: [{df['time_sec'].min():.2f}, {df['time_sec'].max():.2f}]s")

    # Per-run breakdown
    all_runs = sorted(set(r for r, _ in emg_epochs | robot_epochs | fnirs_epochs))
    print(f"\n  Per-run epoch counts:")
    print(f"    {'run_id':<25} {'EMG':>5} {'Robot':>6} {'fNIRS':>6}")
    print(f"    {'-'*25} {'-'*5} {'-'*6} {'-'*6}")
    for run_id in all_runs:
        e = sum(1 for r, _ in emg_epochs if r == run_id)
        r = sum(1 for r, _ in robot_epochs if r == run_id)
        f = sum(1 for r, _ in fnirs_epochs if r == run_id)
        flag = "  ← MISMATCH" if r != e or f != e else ""
        print(f"    {run_id:<25} {e:>5} {r:>6} {f:>6}{flag}")

    return {
        "emg_count": len(emg_epochs),
        "fnirs_count": len(fnirs_epochs),
        "robot_count": len(robot_epochs),
        "common_count": len(common_ef),
        "emg_only": emg_only,
        "fnirs_only": fnirs_only,
    }


# ═══════════════════════════════════════════════════════════════
# 3. RESAMPLING
# ═══════════════════════════════════════════════════════════════

def _target_grid() -> list[float]:
    """100 Hz time grid: [-5.0, -4.99, ..., 14.99, 15.0]"""
    return [EPOCH_TMIN + i / TARGET_HZ for i in range(N_POINTS)]


def resample_stream(
    df: pl.DataFrame,
    time_col: str = "time_sec",
    valid_range: tuple[float, float] | None = None,
) -> pl.DataFrame:
    """
    Resample one stream onto a 100 Hz grid per epoch.

    For each (run_id, task_instance):
      1. Create a uniform 100 Hz time grid from -5 to 15s
      2. Match each grid point to nearest source sample (join_asof)
      3. If valid_range is set, null out data where source time falls
         outside that range (e.g. EMG has no data during -5 to 0s)

    Args:
        df: Source DataFrame with time_col, run_id, task_instance
        time_col: Name of the time column
        valid_range: (tmin, tmax) for valid source times.
                     Values outside are nulled. None = keep all.

    Returns:
        DataFrame on 100 Hz grid (2001 rows per epoch)
    """
    target_times = _target_grid()

    # Cast task_instance to Int64 to avoid dtype mismatch across streams
    df = df.with_columns(pl.col("task_instance").cast(pl.Int64))

    epochs = df.select("run_id", "task_instance").unique()

    # Build target: one row per (epoch × time_point)
    target_parts = []
    for epoch in epochs.iter_rows(named=True):
        target_parts.append(pl.DataFrame({
            "time_sec": target_times,
            "run_id": epoch["run_id"],
            "task_instance": epoch["task_instance"],
        }))
    target = pl.concat(target_parts).with_columns(
        pl.col("task_instance").cast(pl.Int64)
    )

    # Tag source time for range checking (rename to avoid join conflict)
    source = df.sort(time_col).rename({time_col: "__src_time"})

    # Nearest-neighbor match within each epoch group
    resampled = target.join_asof(
        source,
        left_on="time_sec",
        right_on="__src_time",
        by=["run_id", "task_instance"],
        strategy="nearest",
    )

    # Null out data where TARGET time is outside the stream's valid range.
    # e.g. EMG has no data from -5 to 0s, so null those rows even though
    # join_asof matched them to the nearest source point at t=0.
    if valid_range is not None:
        tmin, tmax = valid_range
        mask = (pl.col("time_sec") >= tmin) & (pl.col("time_sec") <= tmax)

        data_cols = [
            c for c in resampled.columns
            if c not in ("run_id", "task_instance", "time_sec", "__src_time")
        ]
        resampled = resampled.with_columns(
            pl.when(mask).then(pl.col(c)).otherwise(None).alias(c)
            for c in data_cols
        )

    return resampled.drop("__src_time")


# ═══════════════════════════════════════════════════════════════
# 4. COMBINE 100 HZ
# ═══════════════════════════════════════════════════════════════

def combine_100hz(
    emg_df: pl.DataFrame,
    robot_df: pl.DataFrame,
    fnirs_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Resample all streams to 100 Hz and join into a single file.

    fNIRS is the base (widest time range: -5 to 15s).
    EMG (0–15s) and robot (0–15s) are left-joined onto fNIRS.
    Baseline period (-5 to 0s) is null for EMG and robot.
    """
    print("  fNIRS (10 Hz → 100 Hz)...")
    fnirs_100 = resample_stream(fnirs_df, valid_range=(-5.0, 15.0))

    print("  EMG (~1926 Hz → 100 Hz)...")
    emg_100 = resample_stream(emg_df, valid_range=(-5.0, 15.0))

    print("  Robot (variable → 100 Hz)...")
    robot_100 = resample_stream(robot_df, valid_range=(-5.0, 15.0))

    # ── Column selection ──
    join_keys = ["run_id", "task_instance", "time_sec"]

    # fNIRS: metadata + HbO/HbR channels
    fnirs_meta = [c for c in ["participant", "is_robot", "task"] if c in fnirs_100.columns]
    fnirs_data = [c for c in fnirs_100.columns if "_hbo" in c or "_hbr" in c]

    # EMG: sensor channels (exclude metadata already in fNIRS)
    emg_exclude = set(join_keys + ["epoch_start", "participant", "is_robot"])
    emg_data = [c for c in emg_100.columns if c not in emg_exclude]

    # Robot: data columns (exclude redundant metadata)
    robot_exclude = set(join_keys + [
        "task_id", "participant", "is_robot", "marker", "marker_repeat_index",
        "sequence_start_timestamp", "duration_seconds", "fnirs_epoch",
    ])
    robot_data = [c for c in robot_100.columns if c not in robot_exclude]

    # ── Build combined file ──
    # fNIRS base → left-join EMG → left-join Robot
    combined = fnirs_100.select(join_keys + fnirs_meta + fnirs_data)

    if emg_data:
        combined = combined.join(
            emg_100.select(join_keys + emg_data), on=join_keys, how="left"
        )
        print(f"    + {len(emg_data)} EMG channels")

    if robot_data:
        combined = combined.join(
            robot_100.select(join_keys + robot_data), on=join_keys, how="left"
        )
        print(f"    + {len(robot_data)} Robot columns")

    print(f"  → {combined.shape[0]:,} rows × {combined.shape[1]} columns")
    return combined


# ═══════════════════════════════════════════════════════════════
# 5. DATA PACKET (full resolution)
# ═══════════════════════════════════════════════════════════════

def export_data_packet(
    emg_df: pl.DataFrame,
    robot_df: pl.DataFrame,
    fnirs_df: pl.DataFrame,
    output_dir: str | Path,
) -> pl.DataFrame:
    """
    Export full-resolution per-stream files + epoch index.

    Creates:
        {output_dir}/emg_full.parquet
        {output_dir}/robot_full.parquet
        {output_dir}/fnirs_full.parquet
        {output_dir}/epoch_index.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emg_df.write_parquet(output_dir / "emg_full.parquet")
    robot_df.write_parquet(output_dir / "robot_full.parquet")
    fnirs_df.write_parquet(output_dir / "fnirs_full.parquet")

    # Build epoch index from fNIRS metadata
    epoch_index = fnirs_df.select([
        "run_id", "task_instance", "participant", "is_robot",
    ]).unique()

    if "task" in fnirs_df.columns:
        task_info = fnirs_df.select([
            "run_id", "task_instance", pl.col("task").alias("task_id"),
        ]).unique()
        epoch_index = epoch_index.join(
            task_info, on=["run_id", "task_instance"], how="left"
        )

    epoch_index = epoch_index.sort("run_id", "task_instance")
    epoch_index.write_csv(output_dir / "epoch_index.csv")

    print(f"  emg_full.parquet:   {emg_df.shape}")
    print(f"  robot_full.parquet: {robot_df.shape}")
    print(f"  fnirs_full.parquet: {fnirs_df.shape}")
    print(f"  epoch_index.csv:    {epoch_index.shape}")

    return epoch_index


# ═══════════════════════════════════════════════════════════════
# 6. POST-BUILD VERIFICATION
# ═══════════════════════════════════════════════════════════════

def verify_output(combined_path: str | Path, output_dir: str | Path) -> None:
    """
    Sanity-check the combined file and data packet after building.

    Prints:
      - Shape, column count, dtypes
      - Per-epoch row count (expect 2001)
      - Baseline vs task null pattern for each stream
      - File sizes
    """
    import os

    combined_path = Path(combined_path)
    output_dir = Path(output_dir)

    df = pl.read_parquet(combined_path)
    print(f"Shape: {df.shape}")
    print(f"Run IDs: {df['run_id'].n_unique()} unique, {df['task_instance'].n_unique()} task instances")

    # Single epoch check
    first_run = df["run_id"][0]
    sample = df.filter((pl.col("run_id") == first_run) & (pl.col("task_instance") == 1))
    print(f"\nEpoch check ({first_run}, task_instance=1):")
    print(f"  Rows: {sample.shape[0]} (expect {N_POINTS})")
    print(f"  time_sec: [{sample['time_sec'].min():.2f}, {sample['time_sec'].max():.2f}]")

    # Baseline vs task nulls
    baseline = sample.filter(pl.col("time_sec") < 0)
    task = sample.filter(pl.col("time_sec") >= 0)

    hbo = next((c for c in sample.columns if "_hbo" in c), None)
    emg = next((c for c in sample.columns if "EMG" in c and "mV" in c), None)
    rob = "timestamp" if "timestamp" in sample.columns else None

    print(f"\nBaseline (-5 to 0s) [{baseline.shape[0]} rows]:")
    if hbo:
        print(f"  {hbo}: {baseline[hbo].null_count()} nulls")
    if emg:
        print(f"  {emg}: {baseline[emg].null_count()} nulls")
    if rob:
        print(f"  {rob}: {baseline[rob].null_count()} nulls")

    print(f"Task (0 to 15s) [{task.shape[0]} rows]:")
    if hbo:
        print(f"  {hbo}: {task[hbo].null_count()} nulls")
    if emg:
        print(f"  {emg}: {task[emg].null_count()} nulls")
    if rob:
        print(f"  {rob}: {task[rob].null_count()} nulls")

    # Overall null percentages
    print(f"\nOverall null %:")
    for group, col in [
        ("fNIRS HbO", hbo),
        ("EMG", emg),
        ("Robot timestamp", rob),
    ]:
        if col and col in df.columns:
            nulls = df[col].null_count()
            print(f"  {group:20s}: {nulls}/{df.shape[0]} ({100*nulls/df.shape[0]:.1f}%)")

    # File sizes
    mb = combined_path.stat().st_size / 1e6
    print(f"\n{combined_path.name}: {mb:.1f} MB")
    dp = output_dir / "data_packet"
    if dp.exists():
        for f in sorted(dp.iterdir()):
            sz = f.stat().st_size / 1e6
            print(f"  {f.name}: {sz:.1f} MB")


# ═══════════════════════════════════════════════════════════════
# 7. TOP-LEVEL
# ═══════════════════════════════════════════════════════════════

def build_combined_export(
    emg_path: str | Path = "data/processed/all_emg_epochs.parquet",
    robot_path: str | Path = "data/processed/robot_first_15s.csv",
    fnirs_path: str | Path = "data/processed/combined_fnirs.csv",
    output_dir: str | Path = "data/processed/combined",
) -> pl.DataFrame:
    """
    One-call export: builds both the 100 Hz combined file and the
    full-resolution data packet.

    Args:
        emg_path:   Path to EMG processed parquet
        robot_path: Path to robot processed CSV
        fnirs_path: Path to fNIRS processed CSV
        output_dir: Directory for all output files

    Returns:
        The 100 Hz combined DataFrame
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──
    print("Loading streams...")
    emg_df = load_emg_epochs(emg_path)
    robot_df = load_robot_epochs(robot_path)
    fnirs_df = load_fnirs_epochs(fnirs_path)
    print(f"  EMG:   {emg_df.shape}")
    print(f"  Robot: {robot_df.shape}")
    print(f"  fNIRS: {fnirs_df.shape}\n")

    # ── Verify ──
    print("Verifying epoch alignment...")
    verify_epoch_alignment(emg_df, robot_df, fnirs_df)
    print()

    # ── 100 Hz combined ──
    print("Building 100 Hz combined file...")
    combined_100hz = combine_100hz(emg_df, robot_df, fnirs_df)
    combined_path = output_dir / "combined_100hz.parquet"
    combined_100hz.write_parquet(combined_path)
    print(f"  Saved: {combined_path}\n")

    # ── Data packet ──
    print("Exporting data packet...")
    export_data_packet(emg_df, robot_df, fnirs_df, output_dir / "data_packet")

    # ── Verify ──
    print("\nVerifying output...")
    verify_output(combined_path, output_dir)

    print("\nDone!")
    return combined_100hz


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine EMG, Robot, and fNIRS data streams"
    )
    parser.add_argument(
        "--emg", default="data/processed/all_emg_epochs.parquet",
        help="Path to EMG processed parquet"
    )
    parser.add_argument(
        "--robot", default="data/processed/robot_first_15s.csv",
        help="Path to robot processed CSV"
    )
    parser.add_argument(
        "--fnirs", default="data/processed/combined_fnirs.csv",
        help="Path to fNIRS processed CSV"
    )
    parser.add_argument(
        "--output", default="data/processed/combined",
        help="Output directory"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Skip building, just verify an existing combined file"
    )
    args = parser.parse_args()

    if args.verify_only:
        verify_output(
            Path(args.output) / "combined_100hz.parquet",
            Path(args.output),
        )
    else:
        build_combined_export(args.emg, args.robot, args.fnirs, args.output)
