import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import csv
    from pathlib import Path
    import polars as pl
    import sys
    sys.path.insert(0, "/Users/haider/code/Franka-fNIRS-EMG-data_analysis")
    from src.loaders.loader import load_data
    import marimo as mo
    import matplotlib.pyplot as plt

    return Path, csv, load_data, mo, pl


@app.cell
def _(load_data, mo):
    DATA_DIR = "data"
    RAW_DIR = f"{DATA_DIR}/raw"
    PROCESSED_DIR = f"{DATA_DIR}/processed"
    FIGURES_DIR = "../figures"

    data_files = load_data(RAW_DIR)
    print(data_files)

    mo.md(f"**Found {len(data_files)} participants:** `{list(data_files.keys())}`")
    return PROCESSED_DIR, RAW_DIR


@app.cell(hide_code=True)
def _(Path, csv, pl):
    def load_trigno_csv(filepath):
        """Load a Delsys Trigno CSV file into main signal + marker DataFrames.

        The Trigno export uses a multi-row header format:
            - 2 rows above the main header contain sensor names
            - The main header row has channel labels (e.g. "Time Series (s)")
            - Data starts 3 rows below the main header
            - A marker table may follow the main data block

        Args:
            filepath: Path to the Trigno CSV file.

        Returns:
            Tuple of (main_df, marker_df):
                main_df:   Polars DataFrame with float sensor columns
                           named "{sensor} | {channel}".
                marker_df: Polars DataFrame with marker events, or empty
                           if no markers found.
        """
        filepath = Path(filepath)

        with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
            rows = [[cell.strip() for cell in row] for row in csv.reader(f)]

        # Find main signal header row
        main_header_idx = None
        for i, row in enumerate(rows):
            if sum("Time Series" in cell for cell in row) >= 2:
                main_header_idx = i
                break

        if main_header_idx is None:
            raise ValueError("Could not find the main signal header row.")

        if main_header_idx < 2:
            raise ValueError("Could not find the sensor-name row above the main header.")

        sensor_row = rows[main_header_idx - 2]
        main_header_row = rows[main_header_idx]

        # Find marker table header row and where it starts
        marker_header_idx = None
        marker_start_idx = None
        marker_signature = ["Type", "Name", "Label", "Time (s)"]

        for i, row in enumerate(rows):
            for j in range(len(row) - len(marker_signature) + 1):
                if row[j:j + len(marker_signature)] == marker_signature:
                    marker_header_idx = i
                    marker_start_idx = j
                    break
            if marker_header_idx is not None:
                break

        # Main columns: everything before marker block
        main_end_idx = marker_start_idx if marker_start_idx is not None else len(main_header_row)

        current_sensor = None
        raw_main_columns = []
        main_indices = []

        for j in range(main_end_idx):
            sensor_cell = sensor_row[j] if j < len(sensor_row) else ""
            header_cell = main_header_row[j] if j < len(main_header_row) else ""

            if sensor_cell:
                current_sensor = sensor_cell

            if header_cell:
                col_name = f"{current_sensor} | {header_cell}" if current_sensor else header_cell
                raw_main_columns.append(col_name)
                main_indices.append(j)

        # Make main columns unique
        counts = {}
        main_columns = []
        for name in raw_main_columns:
            counts[name] = counts.get(name, 0) + 1
            main_columns.append(name if counts[name] == 1 else f"{name}__{counts[name]}")

        # Main data starts after main header + frequency row + delta-t row
        main_data_start_idx = main_header_idx + 3

        main_rows = []
        for row in rows[main_data_start_idx:]:
            if not any(cell != "" for cell in row):
                continue

            values = [row[j] if j < len(row) else "" for j in main_indices]

            # keep only rows that actually contain main-table data
            if any(v != "" for v in values):
                main_rows.append(values)

        main_df = pl.DataFrame(main_rows, schema=main_columns, orient="row")

        # Safely cast main table to floats
        main_df = main_df.with_columns(
            [
                pl.when(pl.col(c).cast(pl.String).str.strip_chars() == "")
                .then(None)
                .otherwise(pl.col(c).cast(pl.String).str.strip_chars())
                .cast(pl.Float64, strict=False)
                .alias(c)
                for c in main_df.columns
            ]
        )

        # Marker table
        if marker_header_idx is None:
            marker_df = pl.DataFrame()
        else:
            marker_header_row = rows[marker_header_idx]
            marker_indices = [j for j in range(marker_start_idx, len(marker_header_row)) if marker_header_row[j] != ""]
            raw_marker_columns = [marker_header_row[j] for j in marker_indices]

            counts = {}
            marker_columns = []
            for name in raw_marker_columns:
                counts[name] = counts.get(name, 0) + 1
                marker_columns.append(name if counts[name] == 1 else f"{name}__{counts[name]}")

            marker_rows = []
            for row in rows[marker_header_idx + 1:]:
                if not any(cell != "" for cell in row):
                    continue

                values = [row[j] if j < len(row) else "" for j in marker_indices]

                if any(v != "" for v in values):
                    marker_rows.append(values)

            marker_df = pl.DataFrame(marker_rows, schema=marker_columns, orient="row")

        return main_df, marker_df

    return (load_trigno_csv,)


@app.cell(hide_code=True)
def _(pl):
    def sync_sensor_streams(df, ref_col=None, strategy="nearest"):
        """
        Sync all sensor channels onto a common timebase.

        Each sensor channel has a paired timestamp column (ending in "Time Series (s)")
        and a value column. This function aligns all value columns onto the timestamps
        of a reference channel using join_asof.

        Args:
            df: Polars DataFrame with interleaved multi-sensor data.
            ref_col: Reference timestamp column name. If None, auto-selects the
                     sensor with the highest sampling rate (most non-null timestamps).
            strategy: join_asof strategy — "nearest" (default), "forward", or "backward".

        Returns:
            Polars DataFrame with shape (n_ref_timestamps, n_value_channels + 1).
            First column is "time" (the reference timestamps).
            Value columns are aligned to that timebase via nearest-neighbor matching.
        """
        # --- Discover channel pairs ---
        ts_cols = [c for c in df.columns if "Time Series" in c]
        val_cols = [c for c in df.columns if "Time Series" not in c]

        pairs = []
        for vc in val_cols:
            prefix = vc.rsplit(" | ", 1)[0]
            channel_name = vc.split(" | ", 1)[1]
            ts_match = (
                f"{prefix} | {channel_name.rsplit(' (', 1)[0]} Time Series (s)"
            )
            if ts_match in ts_cols:
                pairs.append((ts_match, vc))
            else:
                for tc in ts_cols:
                    if prefix in tc and channel_name.split(" (")[0] in tc:
                        pairs.append((tc, vc))
                        break

        if not pairs:
            raise ValueError("No timestamp/value channel pairs found.")

        # --- Select reference timebase ---
        if ref_col is None:
            # Pick the timestamp column with the most non-null values (fastest sensor)
            ref_col = max(
                ts_cols, key=lambda c: df.select(pl.col(c).drop_nulls()).height
            )
            print(f"Auto-selected reference: {ref_col}")

        # Build reference timebase: unique, sorted, non-null timestamps
        ref_df = (
            df.select(ref_col)
            .drop_nulls()
            .rename({ref_col: "time"})
            .sort("time")
            .unique(subset=["time"])
        )
        n_ref = len(ref_df)
        print(f"Reference timebase: {n_ref} timestamps")

        # --- Align each channel ---
        synced_dfs = [ref_df]

        for ch_ts, ch_val in pairs:
            if ch_ts == ref_col:
                # Reference channel — just take the values as-is
                synced_dfs.append(df.select(ch_val))
                continue

            # Extract non-null (time, value) pairs, sort by time
            ch_data = (
                df.select(ch_ts, ch_val)
                .drop_nulls()
                .rename({ch_ts: "time", ch_val: "value"})
                .sort("time")
            )

            # Align to reference timebase
            aligned = (
                ref_df.join_asof(
                    ch_data,
                    on="time",
                    strategy=strategy,
                )
                .select("value")
                .rename({"value": ch_val})
            )

            synced_dfs.append(aligned)

        return pl.concat(synced_dfs, how="horizontal")

    return (sync_sensor_streams,)


@app.cell(hide_code=True)
def _(pl):
    def handle_dropouts(df):
        """
        Handle sensor dropout zeros with a hybrid strategy.

        EMG channels (mV, %): replace 0 with NaN — honest about missing data.
        IMU channels (G, deg/s): replace 0 with NaN, then interpolate —
        the signal is physically smooth over short gaps.

        Args:
            df: Polars DataFrame from sync_sensor_streams (time + value columns).

        Returns:
            Polars DataFrame with the same shape. EMG channels have nulls where
            zeros were; IMU channels have interpolated values.
        """
        emg_cols = [c for c in df.columns if "(mV)" in c or "(%)" in c]
        imu_cols = [c for c in df.columns if "(G)" in c or "(deg/s)" in c]
        value_cols = emg_cols + imu_cols

        # Step 1: replace all 0s with null
        result = df.with_columns(
            pl.when(pl.col(c) == 0).then(None).otherwise(pl.col(c)).alias(c)
            for c in value_cols
        )

        # Step 2: interpolate IMU channels only
        result = result.with_columns(
            pl.col(c).interpolate().alias(c) for c in imu_cols
        )

        return result

    return (handle_dropouts,)


@app.cell
def _(pl):

    def generate_epochs(cleaned, marker_df, epoch_duration=40.0, total_markers=30):
        """
        Generate evenly-spaced epoch markers starting from the first marker.

        Args:
            cleaned: cleaned dataframe with 'time' column
            marker_df: marker dataframe from load_trigno_csv
            epoch_duration: seconds between markers (default: 40)
            total_markers: maximum number of markers (default: 30)

        Returns:
            Polars DataFrame with columns: epoch_id (1-indexed), epoch_start
        """
        first_marker_time = float(marker_df["Time (s)"].to_list()[0])
        time_end = cleaned["time"][-1]

        onsets = []
        t = first_marker_time
        while len(onsets) < total_markers and t + epoch_duration <= time_end:
            onsets.append(t)
            t += epoch_duration

        return pl.DataFrame({
            "epoch_id": list(range(1, len(onsets) + 1)),
            "epoch_start": onsets,
        })

    return (generate_epochs,)


@app.cell(hide_code=True)
def _(pl):

    def add_metadata(epochs_df, run_id):
        """Attach participant and condition metadata to an epochs DataFrame.

        Parses the run_id string (e.g. "sam_robot_1") into participant name
        and robot condition, adding these as columns alongside task_instance
        derived from epoch_id.

        Args:
            epochs_df: DataFrame with an "epoch_id" column (1-indexed).
            run_id:    Session identifier like "sam_robot_1" or "clarence_norobot".

        Returns:
            DataFrame with columns: task_instance, run_id, is_robot, participant.
        """
        import re
        match = re.match(r"^(\w+?)_(robot|norobot)(?:_(\d+))?$", run_id)
        if not match:
            raise ValueError(f"Could not parse run_id: {run_id}")
        participant = match.group(1)
        is_robot = match.group(2) == "robot"
        return epochs_df.with_columns([
            pl.col("epoch_id").alias("task_instance"),
            pl.lit(run_id).alias("run_id"),
            pl.lit(is_robot).alias("is_robot"),
            pl.lit(participant).alias("participant"),
        ]).drop("epoch_id")


    return (add_metadata,)


@app.function(hide_code=True)
def merge_metadata(cleaned, metadata_df):
    """Join epoch metadata onto cleaned signal data via nearest-neighbor time match.

    Uses join_asof to assign each sample to its enclosing epoch based on
    the nearest epoch_start timestamp. Rows before the first epoch are dropped.

    Args:
        cleaned:      Signal DataFrame with a "time" column.
        metadata_df:  Epoch metadata with "epoch_start" and "task_instance" columns.

    Returns:
        DataFrame with signal data plus task_instance, run_id, is_robot, participant.
    """
    result = cleaned.join_asof(
        metadata_df.sort("epoch_start"),
        left_on="time",
        right_on="epoch_start",
        strategy="nearest",
    )
    result = result.drop_nulls(subset=["task_instance"])
    return result


@app.cell(hide_code=True)
def _(pl):

    def filter_epoch_window(merged_df, window_duration=15.0):
        """Trim each epoch to [-5s, +window_duration) around epoch_start.

        Keeps only samples within the analysis window and drops any
        calibrated/auxiliary columns that leaked through from the raw data.

        Args:
            merged_df:       DataFrame with "time" and "epoch_start" columns.
            window_duration: Post-stimulus window length in seconds (default 15).

        Returns:
            Filtered DataFrame with calibrated columns removed.
        """
        result = merged_df.filter(
            (pl.col("time") >= pl.col("epoch_start") - 5.0) & 
            (pl.col("time") < pl.col("epoch_start") + window_duration)
        )
        cal_cols = [c for c in result.columns if "calibrated" in c.lower()]
        if cal_cols:
            result = result.drop(cal_cols)
        return result


    return (filter_epoch_window,)


@app.cell(hide_code=True)
def _(
    PROCESSED_DIR,
    RAW_DIR,
    add_metadata,
    filter_epoch_window,
    generate_epochs,
    handle_dropouts,
    load_data,
    load_trigno_csv,
    pl,
    sync_sensor_streams,
):
    def process_session(_emg_path, _run_id):
        """Run the full EMG processing pipeline for one session.

        Steps: load CSV → sync sensors → handle dropouts → generate epochs
        → attach metadata → merge onto signal → window to [-5, +15s).
        """
        _main_df, _marker_df = load_trigno_csv(_emg_path)
        _synced = sync_sensor_streams(_main_df)
        _cleaned = handle_dropouts(_synced)
        _epochs = generate_epochs(_cleaned, _marker_df)
        _metadata = add_metadata(_epochs, _run_id)
        _merged = merge_metadata(_cleaned, _metadata)
        _windowed = filter_epoch_window(_merged)
        return _windowed

    # Two-pass batch processing:
    #   Pass 1: Run every session through the pipeline to discover the
    #           intersection of columns (different sessions may have
    #           different sensor counts).
    #   Pass 2: Re-run with only common columns, then concatenate.
    _emg_files = load_data(RAW_DIR)

    # First pass: find common columns through full pipeline
    print("Scanning schemas...")
    _all_col_sets = []
    for _run_id in sorted(_emg_files.keys()):
        _emg_path = _emg_files[_run_id]["emg"]
        if _emg_path is None:
            raise ValueError(f"No EMG file found for {_run_id}")
        _df = process_session(_emg_path, _run_id)
        _all_col_sets.append(set(_df.columns))

    _common_cols = _all_col_sets[0]
    for _cols in _all_col_sets[1:]:
        _common_cols &= _cols
    _common_cols_sorted = sorted(_common_cols)
    print(f"Common columns: {len(_common_cols_sorted)}")

    # Second pass: process, align columns, collect
    _all_session_dfs = []
    for _run_id in sorted(_emg_files.keys()):
        _emg_path = _emg_files[_run_id]["emg"]
        if _emg_path is None:
            raise ValueError(f"No EMG file found for {_run_id}")
        print(f"Processing {_run_id}...", end=" ")
        _session_df = process_session(_emg_path, _run_id)
        # Align to common columns
        _session_df = _session_df.select(_common_cols_sorted)
        _n_rows = _session_df.shape[0]
        _n_epochs = _session_df["task_instance"].n_unique()
        print(f"{_n_rows:,} rows, {_n_epochs} epochs")
        _all_session_dfs.append(_session_df)

    # Combine and export
    print()
    print(f"Concatenating {len(_all_session_dfs)} sessions...")
    final_df = pl.concat(_all_session_dfs)
    print(f"Total: {final_df.shape[0]:,} rows x {final_df.shape[1]} columns")

    import os
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    export_path = f"{PROCESSED_DIR}/all_emg_epochs.parquet"
    final_df.write_parquet(export_path)
    print(f"Exported to {export_path}")
    final_df
    return


if __name__ == "__main__":
    app.run()
