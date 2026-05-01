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
    main_df
    return


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


@app.cell
def _(synced_df):
    synced_df
    return


@app.cell(hide_code=True)
def _(pl, synced_df):
    # Count zeros (0, 0.0, or "0") per column in synced_df
    print(f"{'Column':<60} {'Zeros':>8}")
    print("-" * 70)
    total_with_zeros = 0
    for c in synced_df.columns:
        s = synced_df.select(pl.col(c)).to_series()
        numeric_zeros = (s == 0).sum()
        str_zeros = (s.cast(pl.Utf8).str.strip_chars() == "0").sum()
        zeros = numeric_zeros + str_zeros
        if zeros > 0:
            print(f"{c:<60} {zeros:>8}")
            total_with_zeros += 1
    print("-" * 70)
    print(f"Total columns with zeros: {total_with_zeros}")
    return


@app.cell(hide_code=True)
def _(pl, synced_df):
    # Slice 10 rows before and after the first zero in any value column
    _val_cols = [c for c in synced_df.columns if c != "time"]

    _nz_series = synced_df.select(
        pl.sum_horizontal(
            [(pl.col(c) == 0).cast(pl.Int32) for c in _val_cols]
        ).alias("n_zeros")
    ).to_series()

    _first = (_nz_series > 0).arg_true()[0]
    _start = max(0, _first - 10)
    _end = min(len(synced_df), _first + 11)

    # Build a clean slice with a row index and zero count
    synced_df[_start:_end].with_columns(
        pl.lit(list(range(_start, _end))).alias("row"),
        _nz_series[_start:_end].alias("n_zeros"),
    ).select("row", "time", "n_zeros", pl.all().exclude("row", "time", "n_zeros"))
    return


@app.cell
def _(synced_df):


    import plotly.express as px

    quick_plot = lambda s: px.line(y=s.to_list(), title=s.name).show()

    s_ = synced_df["Avanti Sensor 1 (82703) | EMG 1 (mV)"]
    quick_plot(s_[::10])
    return


if __name__ == "__main__":
    app.run()
