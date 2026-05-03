import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import marimo as mo

    DATA_DIR = "data"
    RAW_DIR = f"{DATA_DIR}/raw"
    PROCESSED_DIR = f"{DATA_DIR}/processed"
    FIGURES_DIR = "../figures"
    return RAW_DIR, mo, pl


@app.cell
def _(RAW_DIR, mo):
    import sys
    sys.path.insert(0, "/Users/haider/code/Franka-fNIRS-EMG-data_analysis")
    from src.loaders.loader import load_data

    data_files = load_data(RAW_DIR)
    print(data_files)

    mo.md(f"**Found {len(data_files)} participants:** `{list(data_files.keys())}`")
    return (data_files,)


@app.function
def process_robot_file(
    run_id, robot_path, base_path="/Users/haider/code/Franka-fNIRS-EMG-data_analysis"
):
    """
    Load a single robot CSV, split marker runs into logical task instances, and
    always return a consistent schema across robot/norobot runs.

    Handles:
    - normal runs -> 1 task
    - doubled runs -> split into 2 sequential tasks
    - final overrun -> forced to 1 task

    Returns a pl.DataFrame with a fixed schema suitable for concatenation.
    """
    import os
    import polars as pl

    value_col = "fnirs_epoch"
    time_col = "timestamp"
    participant = run_id.split("_")[0]
    is_robot = "norobot" not in run_id

    fixed_cols = {
        "timestamp": pl.Float64,
        "fnirs_epoch": pl.Int64,
        "expression_happy_index": pl.Int64,
        "Franka_ee": pl.Utf8,
        "Franka_q": pl.Utf8,
        "Franka_dq": pl.Utf8,
        "Franka_tau_J": pl.Utf8,
        "task_id": pl.Utf8,
        "task_instance": pl.Int64,
        "run_id": pl.Utf8,
        "is_robot": pl.Boolean,
        "participant": pl.Utf8,
        "marker": pl.Int64,
        "marker_repeat_index": pl.Int64,
        "sequence_start_timestamp": pl.Float64,
        "duration_seconds": pl.Float64,
    }

    def empty_result():
        return pl.DataFrame(
            schema=fixed_cols
        )

    if robot_path is None:
        return empty_result()

    if not robot_path.startswith("/"):
        robot_path = f"{base_path}/{robot_path}"

    if not os.path.exists(robot_path) or os.path.getsize(robot_path) == 0:
        return empty_result()

    df = pl.read_csv(robot_path)

    # Ensure required base columns exist
    if time_col not in df.columns or value_col not in df.columns:
        return empty_result()

    # Add any missing expected raw columns as nulls so every run matches schema
    for col_name, dtype in [
        ("expression_happy_index", pl.Int64),
        ("Franka_ee", pl.Utf8),
        ("Franka_q", pl.Utf8),
        ("Franka_dq", pl.Utf8),
        ("Franka_tau_J", pl.Utf8),
    ]:
        if col_name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col_name))

    df = (
        df
        .with_columns(
            pl.col(time_col).cast(pl.Float64),
            pl.col(value_col).cast(pl.Int64),
        )
        .sort(time_col)
    )

    # Tag every raw row with a physical run id
    df = df.with_columns(
        (
            (pl.col(value_col) != pl.col(value_col).shift(1))
            .fill_null(True)
            .cast(pl.Int64)
            .cum_sum()
        ).alias("__run_id")
    )

    # Build physical run summary
    runs = (
        df.group_by("__run_id", maintain_order=True)
        .agg(
            pl.first(value_col).alias("marker"),
            pl.first(time_col).alias("start_timestamp"),
            pl.len().alias("run_len"),
        )
        .filter(pl.col("marker") != 0)
        .with_row_count("__run_idx")
    )

    # If there are no usable markers, return rows with null task metadata
    if runs.height == 0:
        result = df.with_columns(
            [
                pl.lit(None, dtype=pl.Utf8).alias("task_id"),
                pl.lit(None, dtype=pl.Int64).alias("task_instance"),
                pl.lit(run_id).alias("run_id"),
                pl.lit(is_robot).alias("is_robot"),
                pl.lit(participant).alias("participant"),
                pl.lit(None, dtype=pl.Int64).alias("marker"),
                pl.lit(None, dtype=pl.Int64).alias("marker_repeat_index"),
                pl.lit(None, dtype=pl.Float64).alias("sequence_start_timestamp"),
                pl.lit(None, dtype=pl.Float64).alias("duration_seconds"),
            ]
        )
    else:
        normal_run_len = runs["run_len"].median()
        last_run_idx = runs["__run_idx"].max()

        runs_with_repeats = (
            runs.with_columns(
                (
                    (pl.col("run_len") / normal_run_len)
                    .round(0)
                    .cast(pl.Int64)
                    .clip(lower_bound=1)
                ).alias("repeat_count")
            )
            .with_columns(
                pl.when(pl.col("__run_idx") == last_run_idx)
                .then(1)
                .otherwise(pl.col("repeat_count"))
                .alias("repeat_count")
            )
            .with_columns(
                pl.col("start_timestamp").shift(-1).alias("next_run_start_timestamp")
            )
            .with_columns(
                (
                    pl.col("next_run_start_timestamp") - pl.col("start_timestamp")
                ).alias("run_duration_seconds")
            )
        )

        sequence_runs = (
            runs_with_repeats.select(
                "__run_id",
                "__run_idx",
                "marker",
                "start_timestamp",
                "run_len",
                "repeat_count",
                "run_duration_seconds",
                pl.int_ranges(0, pl.col("repeat_count")).alias("__dup"),
            )
            .explode("__dup")
            .with_columns(
                (pl.col("__dup") + 1).alias("marker_repeat_index"),
                (pl.col("run_duration_seconds") / pl.col("repeat_count")).alias(
                    "duration_seconds"
                ),
            )
            .drop("__dup")
            .with_row_count("task_index")
        )

        median_dur = sequence_runs["duration_seconds"].drop_nulls().median()
        sequence_runs = sequence_runs.with_columns(
            pl.when(pl.col("duration_seconds").is_null())
            .then(median_dur)
            .otherwise(pl.col("duration_seconds"))
            .alias("duration_seconds")
        )

        result_dfs = []

        for seq in sequence_runs.iter_rows(named=True):
            run_rows = (
                df.filter(pl.col("__run_id") == seq["__run_id"])
                .sort(time_col)
                .with_row_count("__row_in_run")
            )

            n_rows = run_rows.height
            repeat_count = seq["repeat_count"]
            repeat_idx_zero_based = seq["marker_repeat_index"] - 1

            start_idx = int(round(repeat_idx_zero_based * n_rows / repeat_count))
            end_idx = int(round((repeat_idx_zero_based + 1) * n_rows / repeat_count))

            chunk = run_rows.slice(start_idx, max(0, end_idx - start_idx))

            # Use the actual first timestamp of this chunk, not the original run start.
            # Otherwise repeated markers get the same start_timestamp, pushing the
            # second chunk's relative time past 15s and losing data.
            chunk_first_ts = chunk[time_col][0] if chunk.height > 0 else seq["start_timestamp"]
            chunk_duration = chunk[time_col][-1] - chunk_first_ts if chunk.height > 1 else seq["duration_seconds"]

            chunk = chunk.with_columns(
                [
                    pl.lit(f"task_{seq['marker']}").alias("task_id"),
                    pl.lit(seq["task_index"] + 1).alias("task_instance"),
                    pl.lit(run_id).alias("run_id"),
                    pl.lit(is_robot).alias("is_robot"),
                    pl.lit(participant).alias("participant"),
                    pl.lit(seq["marker"]).alias("marker"),
                    pl.lit(seq["marker_repeat_index"]).alias("marker_repeat_index"),
                    pl.lit(chunk_first_ts).alias("sequence_start_timestamp"),
                    pl.lit(chunk_duration).alias("duration_seconds"),
                ]
            )

            if chunk.height > 0:
                result_dfs.append(chunk)

        result = pl.concat(result_dfs) if result_dfs else empty_result()

    # Drop internals
    drop_cols = [c for c in ["__run_id", "__row_in_run", "__run_idx"] if c in result.columns]
    if drop_cols:
        result = result.drop(drop_cols)

    # Add any missing final columns as nulls
    for col_name, dtype in fixed_cols.items():
        if col_name not in result.columns:
            result = result.with_columns(pl.lit(None, dtype=dtype).alias(col_name))

    ordered_cols = list(fixed_cols.keys())
    result = result.select(ordered_cols).sort("timestamp")

    return result


@app.cell
def _(data_files):
    def build_giant_robot_csv(data_files):
        import polars as pl

        dfs = []

        for run_id, paths in data_files.items():
            robot_path = paths.get("robot")
            print(f"Processing {run_id}...")

            df = process_robot_file(run_id, robot_path)
            n_tasks = df.select("task_instance").drop_nulls().n_unique()
            print(f"  -> {df.height} rows, {n_tasks} task instances, is_robot={df['is_robot'][0] if df.height > 0 else 'unknown'}")

            dfs.append(df)

        giant_df = pl.concat(dfs, rechunk=True)
        print(f"\nTotal: {giant_df.height} rows")
        return giant_df

    robot_df = build_giant_robot_csv(data_files)
    return (robot_df,)


@app.cell
def _(robot_df):
    robot_df
    return


@app.cell
def _(pl, robot_df):
    first_15s_df = (
        robot_df
        .filter(
            (pl.col("sequence_start_timestamp").is_not_null()) &
            ((pl.col("timestamp") - pl.col("sequence_start_timestamp")) >= -5.0) &
            ((pl.col("timestamp") - pl.col("sequence_start_timestamp")) < 15.0)
        )
    )

    first_15s_df
    return (first_15s_df,)


@app.cell
def _(first_15s_df):
    first_15s_df.write_csv("data/processed/robot_first_15s.csv")
    return


@app.cell
def _(first_15s_df, pl):
    duration_check = (
        first_15s_df
        .group_by(["run_id", "task_instance"], maintain_order=True)
        .agg(
            pl.min("timestamp").alias("start_ts"),
            pl.max("timestamp").alias("end_ts"),
            (pl.max("timestamp") - pl.min("timestamp")).alias("kept_duration_s"),
            pl.first("marker").alias("marker"),
        )
    )

    duration_check
    return


if __name__ == "__main__":
    app.run()
