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
    sys.path.insert(0, "/Users/haider/code/data_analysis")
    from src.loaders.loader import load_data

    data_files = load_data(RAW_DIR)
    print(data_files)

    mo.md(f"**Found {len(data_files)} participants:** `{list(data_files.keys())}`")
    return (data_files,)


@app.cell
def _(data_files, pl):
    # Show the complete epoch sequence with transitions
    sample_run = "sam_robot_1"
    sample_path = data_files[sample_run]["robot"]
    import os

    if not sample_path.startswith("/"):
        sample_path = os.path.join("/Users/haider/code/data_analysis", sample_path)

    df = pl.read_csv(sample_path)
    df = df.sort("timestamp")

    # epochs = df.select("fnirs_epoch").to_series().to_list()

    # # Find all transitions
    # transitions = []
    # for i in range(1, len(epochs)):
    #     if epochs[i] != epochs[i - 1]:
    #         transitions.append((i, epochs[i - 1], epochs[i]))

    # print(f"Total transitions: {len(transitions)}")
    # print("\nAll transitions (index, from_epoch, to_epoch):")
    # for t in transitions:
    #     print(f"  Row {t[0]}: {t[1]} → {t[2]}")
    return (df,)


@app.cell
def _(data_files):
    robot_paths = [v["robot"] for k, v in data_files.items() if "_robot" in k and "norobot" not in k]
    norobot_paths = [v["robot"] for k, v in data_files.items() if "norobot" in k]
    return


@app.cell
def _(df, pl):
    # diagonsitcs
    value_col = "fnirs_epoch"
    time_col = "timestamp"
    expected_total_markers = 30

    # Collapse contiguous identical values into physical runs
    runs = (
        df
        .sort(time_col)
        .with_columns(
            (pl.col(value_col) != pl.col(value_col).shift(1))
            .cast(pl.Int64)
            .cum_sum()
            .alias("__run_id")
        )
        .group_by("__run_id", maintain_order=True)
        .agg(
            pl.first(value_col).alias("marker"),
            pl.first(time_col).alias("start_timestamp"),
            pl.len().alias("run_len")
        )
        .filter(pl.col("marker") != 0)
        .with_row_count("run_index")
    )

    # Estimate normal run length
    normal_run_len = runs["run_len"].median()
    last_run_index = runs["run_index"].max()

    # Estimate how many logical markers each physical run represents
    runs_with_repeats = (
        runs
        .with_columns(
            (
                (pl.col("run_len") / normal_run_len)
                .round(0)
                .cast(pl.Int64)
                .clip(lower_bound=1)
            ).alias("repeat_count")
        )
        .with_columns(
            pl.when(pl.col("run_index") == last_run_index)
            .then(1)
            .otherwise(pl.col("repeat_count"))
            .alias("repeat_count")
        )
        .with_columns(
            pl.col("start_timestamp").shift(-1).alias("next_run_start_timestamp")
        )
        .with_columns(
            (pl.col("next_run_start_timestamp") - pl.col("start_timestamp")).alias("run_duration_seconds")
        )
    )



    # Expand physical runs into logical markers
    sequence_durations = (
        runs_with_repeats
        .select(
            "run_index",
            "marker",
            "start_timestamp",
            "run_len",
            "repeat_count",
            "run_duration_seconds",
            pl.int_ranges(0, pl.col("repeat_count")).alias("__dup")
        )
        .explode("__dup")
        .drop("__dup")
        .with_columns(
            (pl.col("run_duration_seconds") / pl.col("repeat_count")).alias("duration_seconds")
        )
        .with_row_count("marker_index")
        .head(expected_total_markers)
    )


    sequence_durations = sequence_durations.with_columns(
        pl.when(pl.col("duration_seconds").is_null())
        .then(
            pl.col("duration_seconds").drop_nulls().median().over(pl.lit(1))
        )
        .otherwise(pl.col("duration_seconds"))
        .alias("duration_seconds")
    )

    print(sequence_durations.select("marker", "duration_seconds"))
    print(f"Estimated normal run length: {normal_run_len}")
    print(sequence_durations)
    print(sequence_durations.select("marker", "duration_seconds"))

    duration_summary = (
        sequence_durations
        .filter(pl.col("duration_seconds").is_not_null())
        .select(
            pl.len().alias("n_sequences_measured"),
            pl.col("duration_seconds").mean().alias("mean_duration_seconds"),
            pl.col("duration_seconds").median().alias("median_duration_seconds"),
            pl.col("duration_seconds").min().alias("min_duration_seconds"),
            pl.col("duration_seconds").max().alias("max_duration_seconds")
        )
    )

    print(duration_summary)
    return


@app.cell
def _(df):
    df.schema
    return


if __name__ == "__main__":
    app.run()
