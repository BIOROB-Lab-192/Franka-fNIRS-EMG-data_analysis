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
    return (main_df,)


@app.cell
def _(main_df):
    main_df.schema
    return


@app.cell(hide_code=True)
def _(main_df):
    # Map each value channel to its corresponding timestamp channel
    value_cols = [c for c in main_df.columns if "Time Series" not in c]
    ts_cols = [c for c in main_df.columns if "Time Series" in c]

    channel_pairs = []
    for vc in value_cols:
        prefix = vc.rsplit(" | ", 1)[0]
        channel_name = vc.split(" | ", 1)[1]
        ts_match = f"{prefix} | {channel_name.rsplit(' (', 1)[0]} Time Series (s)"
        if ts_match in ts_cols:
            channel_pairs.append((ts_match, vc))
        else:
            for tc in ts_cols:
                if prefix in tc and channel_name.split(" (")[0] in tc:
                    channel_pairs.append((tc, vc))
                    break
    return (channel_pairs,)


@app.cell(hide_code=True)
def _(main_df):
    # Step 1: Build the reference timebase from Duo 3 EMG 1 timestamps
    ref_ts_col = "Duo Sensor 3 (78042) | EMG 1 Time Series (s)"
    ref_df = (
        main_df.select(ref_ts_col)
        .drop_nulls()
        .rename({ref_ts_col: "time"})
        .sort("time")
        .unique(subset=["time"])
    )

    print(f"Reference timebase: {len(ref_df)} unique timestamps")
    print(f"Range: {ref_df['time'].min():.3f}s to {ref_df['time'].max():.3f}s")
    print(
        f"Effective rate: {len(ref_df) / (ref_df['time'].max() - ref_df['time'].min()):.1f} Hz"
    )
    return ref_df, ref_ts_col


@app.cell(hide_code=True)
def _(channel_pairs, main_df, pl, ref_df, ref_ts_col):
    # Step 2: Sync all channels onto the reference timebase using join_asof

    synced_dfs = [ref_df]  # start with the reference timestamps

    for ch_ts, ch_val in channel_pairs:
        # Skip the reference channel (Duo 3 EMG 1) — already in ref_df
        if ch_ts == ref_ts_col:
            synced_dfs.append(main_df.select(ch_val))
            continue

        # Extract non-null (timestamp, value) pairs for this channel
        ch_data = (
            main_df.select(ch_ts, ch_val)
            .drop_nulls()
            .rename({ch_ts: "time", ch_val: "value"})
            .sort("time")
        )

        # join_asof: for each reference timestamp, find the nearest channel sample
        aligned = (
            ref_df.join_asof(
                ch_data,
                on="time",
                strategy="nearest",
            )
            .select("value")
            .rename({"value": ch_val})
        )

        synced_dfs.append(aligned)

    # Combine all aligned columns
    synced_df = pl.concat(synced_dfs, how="horizontal")
    print(f"Synced shape: {synced_df.shape}")
    print(synced_df.head(3))
    return (synced_df,)


@app.cell
def _(synced_df):
    synced_df
    return


if __name__ == "__main__":
    app.run()
