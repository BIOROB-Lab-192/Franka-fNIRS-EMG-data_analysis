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
def _(data_files):
    robot_paths = [v["robot"] for k, v in data_files.items() if "_robot" in k and "norobot" not in k]
    norobot_paths = [v["robot"] for k, v in data_files.items() if "norobot" in k]

    task_ids = {
        'task_1': 1, 'task_2': 2, 'task_3': 3, 'task_4': 4, 'task_5': 5,
        'task_6': 6, 'task_7': 7, 'task_8': 8, 'task_9': 9, 'task_10': 10,
    }
    return norobot_paths, robot_paths


@app.cell
def _(pl, robot_paths):
    def data_processing(df, robot):
        df = pl.read_csv(robot_paths[1])

        if not robot:
            df = df.drop(["Franka_ee", "Franka_q", "Franka_dq", "Franka_tau_J"], strict=False)    
        df = df.remove(pl.col("fnirs_epoch") == 0)

        return df

    return (data_processing,)


@app.cell
def _(data_processing, norobot_paths):
    df = data_processing(norobot_paths, False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
