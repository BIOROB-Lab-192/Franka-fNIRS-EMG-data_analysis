import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import csv
    from pathlib import Path
    import polars as pl
    import sys
    sys.path.insert(0, "/Users/haider/code/data_analysis")
    from src.loaders.loader import load_data
    import marimo as mo
    import matplotlib.pyplot as plt

    return Path, csv, load_data, mo, pl, plt


@app.cell
def _(load_data, mo):
    DATA_DIR = "data"
    RAW_DIR = f"{DATA_DIR}/raw"
    PROCESSED_DIR = f"{DATA_DIR}/processed"
    FIGURES_DIR = "../figures"

    data_files = load_data(RAW_DIR)
    print(data_files)

    mo.md(f"**Found {len(data_files)} participants:** `{list(data_files.keys())}`")
    return


@app.cell(hide_code=True)
def _(Path, csv, pl):
    def load_trigno_csv(filepath):
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


@app.cell
def _(load_trigno_csv):
    main_df, marker_df = load_trigno_csv("./data/raw/jiang_norobot1/Trial_6.csv")
    return main_df, marker_df


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


@app.cell
def _(main_df, pl, sync_sensor_streams):
    # Sync all sensor streams onto the fastest timebase (Duo 3 EMG 1 @ 1926 Hz)
    synced_df = sync_sensor_streams(main_df)

    print(f"Synced: {synced_df.shape[0]} rows × {synced_df.shape[1]} columns")
    print(
        f"Time range: {synced_df['time'].min():.3f}s → {synced_df['time'].max():.3f}s"
    )
    print(f"Duration: {synced_df['time'].max() - synced_df['time'].min():.2f}s")
    print(
        f"Nulls: {sum(synced_df.select(pl.col(c).null_count()).item() for c in synced_df.columns)}"
    )
    return (synced_df,)


@app.cell(hide_code=True)
def _(cleaned, pl, synced_df):
    # Compare original vs cleaned at the same dropout (rows 2610–2630)
    _start, _end = 2610, 2630

    _orig = synced_df[_start:_end].with_columns(
        pl.lit(list(range(_start, _end))).alias("row"),
    )
    _clean = cleaned[_start:_end].with_columns(
        pl.lit(list(range(_start, _end))).alias("row"),
    )

    # Focus on Avanti 2 ACC/GYRO (the channels that dropped) + Avanti 2 EMG
    _focus = [
        "row",
        "time",
        "Avanti Sensor 2 (82529) | EMG 1 (mV)",
        "Avanti Sensor 2 (82529) | ACC X (G)",
        "Avanti Sensor 2 (82529) | ACC Y (G)",
        "Avanti Sensor 2 (82529) | ACC Z (G)",
        "Avanti Sensor 2 (82529) | GYRO X (deg/s)",
        "Avanti Sensor 2 (82529) | GYRO Y (deg/s)",
        "Avanti Sensor 2 (82529) | GYRO Z (deg/s)",
    ]

    # Build comparison: orig value | cleaned value for each channel
    parts = [_orig.select("row", "time")]
    for _ch in _focus[2:]:  # skip row, time
        _ch_short = _ch.split(" | ", 1)[1].replace(" (", "_").replace(")", "")
        parts.append(_orig.select(pl.col(_ch).alias(f"{_ch_short}_orig")))
        parts.append(_clean.select(pl.col(_ch).alias(f"{_ch_short}_clean")))

    comparison = pl.concat(parts, how="horizontal")
    comparison
    return


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


@app.cell(hide_code=True)
def _(handle_dropouts, pl, synced_df):
    # Apply dropout handling
    cleaned = handle_dropouts(synced_df)

    # Verify
    _emg_cols = [c for c in synced_df.columns if "(mV)" in c or "(%)" in c]
    _imu_cols = [c for c in synced_df.columns if "(G)" in c or "(deg/s)" in c]

    _emg_zeros = sum(
        (synced_df.select(pl.col(c) == 0).sum().item()) for c in _emg_cols
    )
    _imu_zeros = sum(
        (synced_df.select(pl.col(c) == 0).sum().item()) for c in _imu_cols
    )
    _emg_nulls = sum(
        cleaned.select(pl.col(c).null_count()).item() for c in _emg_cols
    )
    _imu_nulls = sum(
        cleaned.select(pl.col(c).null_count()).item() for c in _imu_cols
    )

    print(f"Shape: {cleaned.shape}")
    print(f"EMG: {_emg_zeros} zeros -> {_emg_nulls} nulls")
    print(f"IMU: {_imu_zeros} zeros -> {_imu_nulls} nulls (interpolated)")
    return (cleaned,)


@app.cell
def _(cleaned, marker_df, plt):

    # Sensor 1 marker verification — 15s before and after first marker
    which_marker = 23
    first_marker_time = float(marker_df["Time (s)"].to_list()[0])+ (40 * which_marker)
    t_min = first_marker_time - 5
    t_max = first_marker_time + 5

    time_data = cleaned["time"]
    mask = (time_data >= t_min) & (time_data <= t_max)
    plot_df = cleaned.filter(mask)
    plot_time = time_data.filter(mask).to_list()

    # Sensor 1 channels
    sensor1 = "Avanti Sensor 1 (82703)"
    emg_col = f"{sensor1} | EMG 1 (mV)"
    gyro_cols = [f"{sensor1} | GYRO {ax} (deg/s)" for ax in ["X", "Y", "Z"]]
    acc_cols = [f"{sensor1} | ACC {ax} (G)" for ax in ["X", "Y", "Z"]]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # EMG
    axes[0].plot(plot_time, plot_df[emg_col].to_list(), linewidth=0.5)
    axes[0].set_ylabel("mV")
    axes[0].set_title("EMG - Sensor 1")
    axes[0].axvline(x=first_marker_time, color="red", linestyle="--", linewidth=1.5, label="Marker")
    axes[0].legend(loc="upper right", fontsize=8)

    # Gyro
    for col in gyro_cols:
        ax_label = col.split(" | ")[1].split(" (")[0]
        axes[1].plot(plot_time, plot_df[col].to_list(), linewidth=0.5, label=ax_label)
    axes[1].set_ylabel("deg/s")
    axes[1].set_title("Gyroscope - Sensor 1")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].axvline(x=first_marker_time, color="red", linestyle="--", linewidth=1.5)

    # Accel
    for col in acc_cols:
        ax_label = col.split(" | ")[1].split(" (")[0]
        axes[2].plot(plot_time, plot_df[col].to_list(), linewidth=0.5, label=ax_label)
    axes[2].set_ylabel("G")
    axes[2].set_title("Accelerometer - Sensor 1")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].axvline(x=first_marker_time, color="red", linestyle="--", linewidth=1.5)

    plt.tight_layout()
    plt.show()

    return


@app.cell
def _(cleaned, marker_df, pl):

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
    epochs_df = generate_epochs(cleaned, marker_df)
    epochs_df

    return (epochs_df,)


@app.cell(hide_code=True)
def _(epochs_df, pl):

    def add_metadata(epochs_df, run_id):
        """
        Add task metadata to epochs dataframe.
    
        Args:
            epochs_df: output from generate_epochs()
            run_id: session key e.g. "jiang_norobot_1"
    
        Returns:
            DataFrame with task_instance, run_id, is_robot, participant columns
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

    metadata_df = add_metadata(epochs_df, "jiang_norobot_1")
    metadata_df

    return (metadata_df,)


@app.cell(hide_code=True)
def _(cleaned, metadata_df, pl):

    def merge_metadata(cleaned, metadata_df):
        """
        Merge metadata into cleaned data, labeling each row with its epoch.
    
        Args:
            cleaned: cleaned dataframe with 'time' column
            metadata_df: output from add_metadata()
    
        Returns:
            Cleaned data with task_instance, run_id, is_robot, participant columns added.
            Rows outside any epoch window are dropped.
        """
        # Add epoch boundaries for filtering
        bounds = metadata_df.with_columns([
            pl.col("epoch_start").shift(-1).alias("epoch_end"),
        ])
        # Last epoch extends to end of data
        last_end = cleaned["time"][-1]
        bounds = bounds.with_columns(
            pl.when(pl.col("epoch_end").is_null())
            .then(pl.lit(last_end))
            .otherwise(pl.col("epoch_end"))
            .alias("epoch_end")
        )
    
        # Assign each row to an epoch using join_asof
        # Join cleaned with epoch starts to find which epoch each row belongs to
        result = cleaned.join_asof(
            metadata_df.rename({"epoch_start": "time"}).sort("time"),
            on="time",
            strategy="forward",
        )
    
        # Drop rows that didn't match (before first epoch)
        result = result.drop_nulls(subset=["task_instance"])
    
        return result

    merged_df = merge_metadata(cleaned, metadata_df)
    print(f"Rows: {merged_df.shape[0]:,}  Columns: {merged_df.shape[1]}")
    print(f"Epochs: {merged_df['task_instance'].n_unique()}")
    merged_df

    return


if __name__ == "__main__":
    app.run()
