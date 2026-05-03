import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import numpy as np
    import json
    import ast

    return ast, mo, np, pl, plt


@app.cell
def _(pl):
    DATA_FILE = "./data/processed/combined/combined_100hz.parquet"

    df = pl.read_parquet(DATA_FILE)
    df
    return (df,)


@app.cell
def _(ast, np):
    def unflattern_HT(flat):
        flat = ast.literal_eval(flat)
        T = np.array(flat).reshape(4, 4)
        T = T.T
        return T

    return


@app.cell(hide_code=True)
def fnirs_baseline_switch(mo):
    fnirs_baseline_switch = mo.ui.switch(label="Baseline correction (-5 to 0s)")
    fnirs_baseline_switch
    return (fnirs_baseline_switch,)


@app.cell(hide_code=True)
def _(df, fnirs_baseline_switch, pl, plt):
    # fNIRS Aggregated: Robot vs No-Robot (Global Channel Average + Savitzky-Golay)
    # Reuses: df, pl, plt, np from existing cells
    from scipy.signal import savgol_filter

    fnirs_TIME_MIN = -5.0
    fnirs_TIME_MAX = 15.0
    fnirs_UM_CONVERSION = 1e6
    fnirs_SG_WINDOW = 101  # 1 second at 100 Hz (must be odd)
    fnirs_SG_ORDER = 3

    # Filter to time window
    fnirs_filtered = df.filter(
        (pl.col("time_sec") >= fnirs_TIME_MIN)
        & (pl.col("time_sec") <= fnirs_TIME_MAX)
    )

    # Identify fNIRS channels
    fnirs_hbo_cols = [c for c in df.columns if c.endswith("_hbo")]
    fnirs_hbr_cols = [c for c in df.columns if c.endswith("_hbr")]

    # Option A: average channels within each run first
    fnirs_run_avg = fnirs_filtered.with_columns(
        [
            pl.mean_horizontal(fnirs_hbo_cols).alias("hbo_mean"),
            pl.mean_horizontal(fnirs_hbr_cols).alias("hbr_mean"),
        ]
    )

    # Baseline correction
    if fnirs_baseline_switch.value:
        for _col in ["hbo_mean", "hbr_mean"]:
            _bl = (
                fnirs_run_avg.filter(pl.col("time_sec") < 0)
                .group_by("run_id")
                .agg(pl.col(_col).mean().alias("_base"))
            )
            fnirs_run_avg = (
                fnirs_run_avg.join(_bl, on="run_id", how="left")
                .with_columns((pl.col(_col) - pl.col("_base")).alias(_col))
                .drop("_base")
            )

    # Average across runs per condition per time point
    fnirs_avg = (
        fnirs_run_avg.group_by("time_sec", "is_robot")
        .agg(
            [
                pl.col("hbo_mean").mean().alias("hbo"),
                pl.col("hbr_mean").mean().alias("hbr"),
            ]
        )
        .sort("time_sec")
    )

    # Convert to μM
    fnirs_avg = fnirs_avg.with_columns(
        [
            (pl.col("hbo") * fnirs_UM_CONVERSION).alias("hbo"),
            (pl.col("hbr") * fnirs_UM_CONVERSION).alias("hbr"),
        ]
    )

    # Split by condition
    fnirs_robot = fnirs_avg.filter(pl.col("is_robot") == True)
    fnirs_no_robot = fnirs_avg.filter(pl.col("is_robot") == False)

    # Apply Savitzky-Golay smoothing
    fnirs_robot_hbo_smooth = savgol_filter(
        fnirs_robot["hbo"], fnirs_SG_WINDOW, fnirs_SG_ORDER
    )
    fnirs_robot_hbr_smooth = savgol_filter(
        fnirs_robot["hbr"], fnirs_SG_WINDOW, fnirs_SG_ORDER
    )
    fnirs_no_robot_hbo_smooth = savgol_filter(
        fnirs_no_robot["hbo"], fnirs_SG_WINDOW, fnirs_SG_ORDER
    )
    fnirs_no_robot_hbr_smooth = savgol_filter(
        fnirs_no_robot["hbr"], fnirs_SG_WINDOW, fnirs_SG_ORDER
    )

    # Plot
    fnirs_fig, (fnirs_ax1, fnirs_ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Robot
    fnirs_ax1.plot(
        fnirs_robot["time_sec"], fnirs_robot_hbo_smooth, label="HbO", color="red"
    )
    fnirs_ax1.plot(
        fnirs_robot["time_sec"], fnirs_robot_hbr_smooth, label="HbR", color="blue"
    )
    fnirs_ax1.set_title("Robot Trials")
    fnirs_ax1.set_xlabel("Time (s)")
    fnirs_ax1.set_ylabel("Concentration (μM)")
    fnirs_ax1.legend()
    fnirs_ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    # No-robot
    fnirs_ax2.plot(
        fnirs_no_robot["time_sec"],
        fnirs_no_robot_hbo_smooth,
        label="HbO",
        color="red",
    )
    fnirs_ax2.plot(
        fnirs_no_robot["time_sec"],
        fnirs_no_robot_hbr_smooth,
        label="HbR",
        color="blue",
    )
    fnirs_ax2.set_title("No-Robot Trials")
    fnirs_ax2.set_xlabel("Time (s)")
    fnirs_ax2.set_ylabel("Concentration (μM)")
    fnirs_ax2.legend()
    fnirs_ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fnirs_fig
    return (savgol_filter,)


@app.cell(hide_code=True)
def _(df, mo, pl):
    # fNIRS Overall Summary Stats
    # Reuses: df, pl, mo from existing cells

    _fnirs_ov_filtered = df.filter(
        (pl.col("time_sec") >= -5.0) & (pl.col("time_sec") <= 15.0)
    )
    _fnirs_ov_hbo = [c for c in df.columns if c.endswith("_hbo")]
    _fnirs_ov_hbr = [c for c in df.columns if c.endswith("_hbr")]

    # Baseline data (time_sec < 0)
    _fnirs_ov_bl = _fnirs_ov_filtered.filter(pl.col("time_sec") < 0)

    _fnirs_ov_lines = []
    for _rob, _cond in [(True, "Robot"), (False, "No-Robot")]:
        _cdf = _fnirs_ov_filtered.filter(pl.col("is_robot") == _rob)
        _runs = _cdf["run_id"].n_unique()
        _parts = _cdf["participant"].n_unique()
        _tasks = _cdf["task"].n_unique()
        _ra = _cdf.with_columns(
            [
                pl.mean_horizontal(_fnirs_ov_hbo).alias("_hbo"),
                pl.mean_horizontal(_fnirs_ov_hbr).alias("_hbr"),
            ]
        )
        _hbo_m = _ra["_hbo"].mean() * 1e6
        _hbo_x = _ra["_hbo"].max() * 1e6
        _hbr_m = _ra["_hbr"].mean() * 1e6
        _hbr_x = _ra["_hbr"].max() * 1e6
        # Baseline: mean during time_sec < 0
        _bl_cdf = _fnirs_ov_bl.filter(pl.col("is_robot") == _rob)
        _bl_ra = _bl_cdf.with_columns(
            [
                pl.mean_horizontal(_fnirs_ov_hbo).alias("_hbo"),
                pl.mean_horizontal(_fnirs_ov_hbr).alias("_hbr"),
            ]
        )
        _hbo_bl = _bl_ra["_hbo"].mean() * 1e6 if _bl_ra.height > 0 else 0.0
        _hbr_bl = _bl_ra["_hbr"].mean() * 1e6 if _bl_ra.height > 0 else 0.0
        _fnirs_ov_lines.append(
            f"| {_cond} | {_runs} | {_parts} | {_tasks} | {_hbo_m:.3f} | {_hbo_x:.3f} | {_hbo_bl:.3f} | {_hbr_m:.3f} | {_hbr_x:.3f} | {_hbr_bl:.3f} |"
        )

    _fnirs_ov_tbl = "| Condition | Runs | Participants | Tasks | HbO Mean (μM) | HbO Max (μM) | HbO Baseline (μM) | HbR Mean (μM) | HbR Max (μM) | HbR Baseline (μM) |\n"
    _fnirs_ov_tbl += "|-----------|------|--------------|-------|---------------|--------------|-------------------|---------------|--------------|-------------------|\n"
    _fnirs_ov_tbl += "\n".join(_fnirs_ov_lines)

    mo.md(f"""
    ### fNIRS Overall Summary

    Global channel average across all tasks, participants, and time points (-5 to 15s).

    {_fnirs_ov_tbl}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # fNIRS Channel Selector Widget
    # Reuses: mo from imports cell

    fnirs_channel_options = {}
    _fnirs_src_det = {
        1: ["D1", "D2", "D3", "D8"],
        2: ["D1", "D3", "D4", "D9"],
        3: ["D2", "D3", "D10"],
        4: ["D3", "D4", "D11"],
        5: ["D5", "D6", "D7", "D12"],
        6: ["D5", "D7", "D13"],
        7: ["D6", "D7", "D14"],
        8: ["D7", "D15"],
    }
    for _s, _dets in _fnirs_src_det.items():
        for _d in _dets:
            for _c in ["HbO", "HbR"]:
                _lbl = f"S{_s}→{_d} {_c}"
                fnirs_channel_options[_lbl] = f"S{_s}_{_d}_{_c.lower()}"

    fnirs_channel_selector = mo.ui.multiselect(
        options=list(fnirs_channel_options.keys()),
        value=list(fnirs_channel_options.keys()),
        label="Select fNIRS channels",
    )
    fnirs_channel_selector
    return fnirs_channel_options, fnirs_channel_selector


@app.cell(hide_code=True)
def _(df, fnirs_channel_options, fnirs_channel_selector, mo, pl):
    # fNIRS Channel Plot Summary Stats
    # Reuses: df, pl, mo, fnirs_channel_selector, fnirs_channel_options from existing cells

    _fnirs_ch_statsFiltered = df.filter(
        (pl.col("time_sec") >= -5.0) & (pl.col("time_sec") <= 15.0)
    )
    _fnirs_ch_bl = _fnirs_ch_statsFiltered.filter(pl.col("time_sec") < 0)

    _fnirs_ch_lines = []
    for _lbl in fnirs_channel_selector.value:
        _col = fnirs_channel_options[_lbl]
        _pair = _lbl.replace(" HbO", "").replace(" HbR", "")
        _chrom = "HbO" if "HbO" in _lbl else "HbR"
        for _rob, _cond in [(True, "Robot"), (False, "No-Robot")]:
            _vals = (
                _fnirs_ch_statsFiltered.filter(pl.col("is_robot") == _rob)[
                    _col
                ].to_numpy()
                * 1e6
            )
            if len(_vals) == 0:
                continue
            _bl_vals = (
                _fnirs_ch_bl.filter(pl.col("is_robot") == _rob)[_col].to_numpy()
                * 1e6
            )
            _bl_mean = _bl_vals.mean() if len(_bl_vals) > 0 else 0.0
            _fnirs_ch_lines.append(
                f"| {_pair} | {_chrom} | {_cond} | {_vals.mean():.3f} | {_vals.max():.3f} | {_bl_mean:.3f} |"
            )

    _fnirs_ch_tbl = (
        "| Channel | Chrom | Condition | Mean (μM) | Max (μM) | Baseline (μM) |\n"
    )
    _fnirs_ch_tbl += (
        "|---------|-------|-----------|-----------|----------|---------------|\n"
    )
    _fnirs_ch_tbl += "\n".join(_fnirs_ch_lines)

    mo.md(f"""
    ### fNIRS Per-Channel Summary

    Mean and max concentration (μM) per source-detector channel, across all runs and time points.

    {_fnirs_ch_tbl}
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(
    df,
    fnirs_baseline_switch,
    fnirs_channel_options,
    fnirs_channel_selector,
    pl,
    plt,
    savgol_filter,
):
    # fNIRS Per-Channel Interactive Plot
    # Reuses: df, pl, plt, np, savgol_filter from existing cells

    fnirs_ch_TIME_MIN = -5.0
    fnirs_ch_TIME_MAX = 15.0
    fnirs_ch_UM = 1e6
    fnirs_ch_SG_WIN = 101
    fnirs_ch_SG_ORD = 3

    # Filter to time window
    fnirs_ch_filtered = df.filter(
        (pl.col("time_sec") >= fnirs_ch_TIME_MIN)
        & (pl.col("time_sec") <= fnirs_ch_TIME_MAX)
    )

    # Get selected channels
    fnirs_ch_selected = [
        fnirs_channel_options[_ch] for _ch in fnirs_channel_selector.value
    ]

    # Baseline correction
    if fnirs_baseline_switch.value:
        for _col in fnirs_ch_selected:
            _bl = (
                fnirs_ch_filtered.filter(pl.col("time_sec") < 0)
                .group_by("run_id")
                .agg(pl.col(_col).mean().alias("_base"))
            )
            fnirs_ch_filtered = (
                fnirs_ch_filtered.join(_bl, on="run_id", how="left")
                .with_columns((pl.col(_col) - pl.col("_base")).alias(_col))
                .drop("_base")
            )

    # Build pair → {HbO: col, HbR: col}
    fnirs_ch_pairs = {}
    for _lbl in fnirs_channel_selector.value:
        _pair = _lbl.replace(" HbO", "").replace(" HbR", "")
        _chrom = "HbO" if "HbO" in _lbl else "HbR"
        fnirs_ch_pairs.setdefault(_pair, {})[_chrom] = fnirs_channel_options[_lbl]

    # No selection case
    if not fnirs_ch_selected:
        fnirs_ch_fig, _ax = plt.subplots(figsize=(14, 5))
        _ax.text(
            0.5, 0.5, "No channels selected", ha="center", va="center", fontsize=14
        )
        _ax.set_axis_off()
        plt.show()

    else:
        fnirs_ch_time = None
        fnirs_ch_robot = {}
        fnirs_ch_no_robot = {}

        for _col in fnirs_ch_selected:
            _avg = (
                fnirs_ch_filtered.group_by("time_sec", "is_robot")
                .agg(pl.col(_col).mean().alias("val"))
                .sort("time_sec")
                .with_columns((pl.col("val") * fnirs_ch_UM).alias("val"))
            )

            if fnirs_ch_time is None:
                fnirs_ch_time = _avg.filter(pl.col("is_robot") == True)[
                    "time_sec"
                ].to_numpy()

            _robot = _avg.filter(pl.col("is_robot") == True)["val"].to_numpy()
            _no_robot = _avg.filter(pl.col("is_robot") == False)["val"].to_numpy()

            if len(_robot) > fnirs_ch_SG_WIN:
                _robot = savgol_filter(_robot, fnirs_ch_SG_WIN, fnirs_ch_SG_ORD)
            if len(_no_robot) > fnirs_ch_SG_WIN:
                _no_robot = savgol_filter(
                    _no_robot, fnirs_ch_SG_WIN, fnirs_ch_SG_ORD
                )

            fnirs_ch_robot[_col] = _robot
            fnirs_ch_no_robot[_col] = _no_robot

        # Colors per pair
        _pair_names = sorted(fnirs_ch_pairs.keys())
        _cmap = plt.cm.tab10 if len(_pair_names) <= 10 else plt.cm.tab20
        fnirs_ch_pair_colors = {
            _p: _cmap(_i / max(len(_pair_names) - 1, 1))
            for _i, _p in enumerate(_pair_names)
        }

        # Create figure (base height = 5)
        fnirs_ch_fig, (fnirs_ch_ax1, fnirs_ch_ax2) = plt.subplots(
            1, 2, figsize=(14, 5)
        )

        # Plot lines
        for _pair, _chroms in fnirs_ch_pairs.items():
            _c = fnirs_ch_pair_colors[_pair]

            if "HbO" in _chroms:
                _col = _chroms["HbO"]
                fnirs_ch_ax1.plot(
                    fnirs_ch_time, fnirs_ch_robot[_col], color=_c, linestyle="-"
                )
                fnirs_ch_ax2.plot(
                    fnirs_ch_time, fnirs_ch_no_robot[_col], color=_c, linestyle="-"
                )

            if "HbR" in _chroms:
                _col = _chroms["HbR"]
                fnirs_ch_ax1.plot(
                    fnirs_ch_time, fnirs_ch_robot[_col], color=_c, linestyle="--"
                )
                fnirs_ch_ax2.plot(
                    fnirs_ch_time,
                    fnirs_ch_no_robot[_col],
                    color=_c,
                    linestyle="--",
                )

        # Axes formatting
        fnirs_ch_ax1.set_title("Robot Trials")
        fnirs_ch_ax1.set_xlabel("Time (s)")
        fnirs_ch_ax1.set_ylabel("Concentration (μM)")
        fnirs_ch_ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        fnirs_ch_ax2.set_title("No-Robot Trials")
        fnirs_ch_ax2.set_xlabel("Time (s)")
        fnirs_ch_ax2.set_ylabel("Concentration (μM)")
        fnirs_ch_ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        fnirs_ch_ax1.set_ylim(-1, 2)
        fnirs_ch_ax2.set_ylim(-1, 2)

        # Legend handles
        from matplotlib.lines import Line2D

        _legend_handles = []
        for _pair in _pair_names:
            _c = fnirs_ch_pair_colors[_pair]
            if "HbO" in fnirs_ch_pairs[_pair]:
                _legend_handles.append(
                    Line2D([0], [0], color=_c, linestyle="-", label=f"{_pair} HbO")
                )
            if "HbR" in fnirs_ch_pairs[_pair]:
                _legend_handles.append(
                    Line2D(
                        [0], [0], color=_c, linestyle="--", label=f"{_pair} HbR"
                    )
                )

        _n_actual = len(_legend_handles)
        _ncols_actual = min(_n_actual, 8)
        _n_legend_rows = (_n_actual + _ncols_actual - 1) // _ncols_actual

        # Grow figure ONLY when needed
        fnirs_ch_fig.set_size_inches(14, 5 + 0.35 * _n_legend_rows)

        # Adjust bottom spacing dynamically
        _bottom = min(0.34, 0.12 + 0.045 * _n_legend_rows)

        fnirs_ch_fig.legend(
            handles=_legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=_ncols_actual,
            fontsize=9,
            frameon=False,
            handlelength=2,
            handletextpad=0.5,
            columnspacing=1.5,
        )

        fnirs_ch_fig.subplots_adjust(
            left=0.06,
            right=0.99,
            top=0.93,
            bottom=_bottom,
            wspace=0.20,
        )

        plt.show()
    return (Line2D,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(df, mo):
    # fNIRS Task Selector Widget
    # Reuses: mo, df from existing cells

    _fnirs_task_sorted = sorted(
        df["task"].unique().to_list(), key=lambda _x: int(_x.split("_")[1])
    )

    fnirs_task_selector = mo.ui.multiselect(
        options=_fnirs_task_sorted, value=_fnirs_task_sorted, label="Select tasks"
    )
    fnirs_task_selector
    return (fnirs_task_selector,)


@app.cell(hide_code=True)
def _(df, fnirs_task_selector, mo, pl):
    # fNIRS Task Plot Summary Stats
    # Reuses: df, pl, mo, fnirs_task_selector from existing cells

    _fnirs_task_statsFiltered = df.filter(
        (pl.col("time_sec") >= -5.0) & (pl.col("time_sec") <= 15.0)
    )
    _fnirs_task_hbo_cols_s = [c for c in df.columns if c.endswith("_hbo")]
    _fnirs_task_hbr_cols_s = [c for c in df.columns if c.endswith("_hbr")]
    _fnirs_task_bl = _fnirs_task_statsFiltered.filter(pl.col("time_sec") < 0)

    _fnirs_task_lines = []
    for _task in fnirs_task_selector.value:
        _tdf = _fnirs_task_statsFiltered.filter(pl.col("task") == _task)
        _bl_tdf = _fnirs_task_bl.filter(pl.col("task") == _task)
        for _rob, _lbl in [(True, "Robot"), (False, "No-Robot")]:
            _cdf = _tdf.filter(pl.col("is_robot") == _rob)
            _runs = _cdf.select("run_id", "task_instance").unique().height
            if _runs == 0:
                continue
            _ra = _cdf.with_columns(
                [
                    pl.mean_horizontal(_fnirs_task_hbo_cols_s).alias("_hbo"),
                    pl.mean_horizontal(_fnirs_task_hbr_cols_s).alias("_hbr"),
                ]
            )
            _hbo_m = _ra["_hbo"].mean() * 1e6
            _hbo_x = _ra["_hbo"].max() * 1e6
            _hbr_m = _ra["_hbr"].mean() * 1e6
            _hbr_x = _ra["_hbr"].max() * 1e6
            # Baseline
            _bl_cdf = _bl_tdf.filter(pl.col("is_robot") == _rob)
            _bl_ra = _bl_cdf.with_columns(
                [
                    pl.mean_horizontal(_fnirs_task_hbo_cols_s).alias("_hbo"),
                    pl.mean_horizontal(_fnirs_task_hbr_cols_s).alias("_hbr"),
                ]
            )
            _hbo_bl = _bl_ra["_hbo"].mean() * 1e6 if _bl_ra.height > 0 else 0.0
            _hbr_bl = _bl_ra["_hbr"].mean() * 1e6 if _bl_ra.height > 0 else 0.0
            _fnirs_task_lines.append(
                f"| {_task} | {_lbl} | {_runs} | {_hbo_m:.3f} | {_hbo_x:.3f} | {_hbo_bl:.3f} | {_hbr_m:.3f} | {_hbr_x:.3f} | {_hbr_bl:.3f} |"
            )

    _fnirs_task_tbl = "| Task | Condition | Runs | HbO Mean (μM) | HbO Max (μM) | HbO Baseline (μM) | HbR Mean (μM) | HbR Max (μM) | HbR Baseline (μM) |\n"
    _fnirs_task_tbl += "|------|-----------|------|---------------|--------------|-------------------|---------------|--------------|-------------------|\n"
    _fnirs_task_tbl += "\n".join(_fnirs_task_lines)

    mo.md(f"""
    ### fNIRS Per-Task Summary

    Mean and max concentration (μM) of the global channel average, across all runs per task and condition.

    {_fnirs_task_tbl}
    """)
    return


@app.cell(hide_code=True)
def _(
    Line2D,
    df,
    fnirs_baseline_switch,
    fnirs_task_selector,
    pl,
    plt,
    savgol_filter,
):
    # fNIRS Per-Task Interactive Plot
    # Reuses: df, pl, plt, np, savgol_filter, Line2D from existing cells

    fnirs_task_TIME_MIN = -5.0
    fnirs_task_TIME_MAX = 15.0
    fnirs_task_UM = 1e6
    fnirs_task_SG_WIN = 101
    fnirs_task_SG_ORD = 3

    # Filter to time window
    fnirs_task_filtered = df.filter(
        (pl.col("time_sec") >= fnirs_task_TIME_MIN)
        & (pl.col("time_sec") <= fnirs_task_TIME_MAX)
    )

    # Identify fNIRS channels
    fnirs_task_hbo_cols = [c for c in df.columns if c.endswith("_hbo")]
    fnirs_task_hbr_cols = [c for c in df.columns if c.endswith("_hbr")]

    # Get selected tasks
    fnirs_task_selected = fnirs_task_selector.value

    if not fnirs_task_selected:
        fnirs_task_fig, _ax = plt.subplots(figsize=(14, 5))
        _ax.text(
            0.5, 0.5, "No tasks selected", ha="center", va="center", fontsize=14
        )
        _ax.set_axis_off()
        plt.show()
    else:
        # Per-task: global mean HbO/HbR across all channels, then across runs
        fnirs_task_time = None
        fnirs_task_robot = {}
        fnirs_task_no_robot = {}

        for _task in fnirs_task_selected:
            _task_df = fnirs_task_filtered.filter(pl.col("task") == _task)

            # Average all channels within each run
            _task_run_avg = _task_df.with_columns(
                [
                    pl.mean_horizontal(fnirs_task_hbo_cols).alias("hbo_mean"),
                    pl.mean_horizontal(fnirs_task_hbr_cols).alias("hbr_mean"),
                ]
            )

            # Baseline correction
            if fnirs_baseline_switch.value:
                for _col in ["hbo_mean", "hbr_mean"]:
                    _bl = (
                        _task_run_avg.filter(pl.col("time_sec") < 0)
                        .group_by("run_id")
                        .agg(pl.col(_col).mean().alias("_base"))
                    )
                    _task_run_avg = (
                        _task_run_avg.join(_bl, on="run_id", how="left")
                        .with_columns((pl.col(_col) - pl.col("_base")).alias(_col))
                        .drop("_base")
                    )

            # Average across runs per condition per time point
            _task_avg = (
                _task_run_avg.group_by("time_sec", "is_robot")
                .agg(
                    [
                        pl.col("hbo_mean").mean().alias("hbo"),
                        pl.col("hbr_mean").mean().alias("hbr"),
                    ]
                )
                .sort("time_sec")
                .with_columns(
                    [
                        (pl.col("hbo") * fnirs_task_UM).alias("hbo"),
                        (pl.col("hbr") * fnirs_task_UM).alias("hbr"),
                    ]
                )
            )

            if fnirs_task_time is None:
                fnirs_task_time = _task_avg.filter(pl.col("is_robot") == True)[
                    "time_sec"
                ].to_numpy()

            _robot_hbo = _task_avg.filter(pl.col("is_robot") == True)[
                "hbo"
            ].to_numpy()
            _robot_hbr = _task_avg.filter(pl.col("is_robot") == True)[
                "hbr"
            ].to_numpy()
            _no_robot_hbo = _task_avg.filter(pl.col("is_robot") == False)[
                "hbo"
            ].to_numpy()
            _no_robot_hbr = _task_avg.filter(pl.col("is_robot") == False)[
                "hbr"
            ].to_numpy()

            if len(_robot_hbo) > fnirs_task_SG_WIN:
                _robot_hbo = savgol_filter(
                    _robot_hbo, fnirs_task_SG_WIN, fnirs_task_SG_ORD
                )
                _robot_hbr = savgol_filter(
                    _robot_hbr, fnirs_task_SG_WIN, fnirs_task_SG_ORD
                )
            if len(_no_robot_hbo) > fnirs_task_SG_WIN:
                _no_robot_hbo = savgol_filter(
                    _no_robot_hbo, fnirs_task_SG_WIN, fnirs_task_SG_ORD
                )
                _no_robot_hbr = savgol_filter(
                    _no_robot_hbr, fnirs_task_SG_WIN, fnirs_task_SG_ORD
                )

            fnirs_task_robot[_task] = {"hbo": _robot_hbo, "hbr": _robot_hbr}
            fnirs_task_no_robot[_task] = {
                "hbo": _no_robot_hbo,
                "hbr": _no_robot_hbr,
            }

        # Color: one unique color per task
        _n_tasks = len(fnirs_task_selected)
        _cmap = plt.cm.tab10 if _n_tasks <= 10 else plt.cm.tab20
        fnirs_task_colors = {
            _t: _cmap(_i / max(_n_tasks - 1, 1))
            for _i, _t in enumerate(fnirs_task_selected)
        }

        fnirs_task_fig, (fnirs_task_ax1, fnirs_task_ax2) = plt.subplots(
            1, 2, figsize=(14, 5)
        )

        for _task in fnirs_task_selected:
            _c = fnirs_task_colors[_task]
            fnirs_task_ax1.plot(
                fnirs_task_time,
                fnirs_task_robot[_task]["hbo"],
                color=_c,
                linestyle="-",
            )
            fnirs_task_ax2.plot(
                fnirs_task_time,
                fnirs_task_no_robot[_task]["hbo"],
                color=_c,
                linestyle="-",
            )
            fnirs_task_ax1.plot(
                fnirs_task_time,
                fnirs_task_robot[_task]["hbr"],
                color=_c,
                linestyle="--",
            )
            fnirs_task_ax2.plot(
                fnirs_task_time,
                fnirs_task_no_robot[_task]["hbr"],
                color=_c,
                linestyle="--",
            )

        fnirs_task_ax1.set_title("Robot Trials")
        fnirs_task_ax1.set_xlabel("Time (s)")
        fnirs_task_ax1.set_ylabel("Concentration (μM)")
        fnirs_task_ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        fnirs_task_ax1.set_ylim(-1, 2)

        fnirs_task_ax2.set_title("No-Robot Trials")
        fnirs_task_ax2.set_xlabel("Time (s)")
        fnirs_task_ax2.set_ylabel("Concentration (μM)")
        fnirs_task_ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        fnirs_task_ax2.set_ylim(-1, 2)

        # Legend handles
        _legend_handles = []
        for _task in fnirs_task_selected:
            _c = fnirs_task_colors[_task]
            _legend_handles.append(
                Line2D([0], [0], color=_c, linestyle="-", label=f"{_task} HbO")
            )
            _legend_handles.append(
                Line2D([0], [0], color=_c, linestyle="--", label=f"{_task} HbR")
            )

        _n_actual = len(_legend_handles)
        _ncols_actual = min(_n_actual, 8)
        _n_legend_rows = (_n_actual + _ncols_actual - 1) // _ncols_actual

        # Grow figure ONLY when needed
        fnirs_task_fig.set_size_inches(14, 5 + 0.35 * _n_legend_rows)

        # Adjust bottom spacing dynamically
        _bottom = min(0.34, 0.12 + 0.045 * _n_legend_rows)

        fnirs_task_fig.legend(
            handles=_legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=_ncols_actual,
            fontsize=9,
            frameon=False,
            handlelength=2,
            handletextpad=0.5,
            columnspacing=1.5,
        )

        fnirs_task_fig.subplots_adjust(
            left=0.06,
            right=0.99,
            top=0.93,
            bottom=_bottom,
            wspace=0.20,
        )

        plt.show()
    return


@app.cell(hide_code=True)
def emg_baseline_switch(mo):
    emg_baseline_switch = mo.ui.switch(label="Baseline correction (-5 to 0s)")
    emg_baseline_switch
    return (emg_baseline_switch,)


@app.cell(hide_code=True)
def _(df, emg_baseline_switch, pl, plt):
    # EMG Aggregated: Robot vs No-Robot (Global Sensor Average)
    # Reuses: df, pl, plt, np from existing cells

    emg_ov_TIME_MIN = -5.0
    emg_ov_TIME_MAX = 15.0

    emg_ov_cols = [c for c in df.columns if "EMG" in c and c.endswith("(mV)")]

    emg_ov_filtered = df.filter(
        (pl.col("time_sec") >= emg_ov_TIME_MIN)
        & (pl.col("time_sec") <= emg_ov_TIME_MAX)
    )

    emg_ov_run_avg = emg_ov_filtered.with_columns(
        pl.mean_horizontal(emg_ov_cols).alias("emg_mean")
    )

    # Baseline correction
    if emg_baseline_switch.value:
        for _col in emg_ov_cols:
            _bl = (
                emg_ov_run_avg.filter(pl.col("time_sec") < 0)
                .group_by("run_id")
                .agg(pl.col(_col).mean().alias("_base"))
            )
            emg_ov_run_avg = (
                emg_ov_run_avg.join(_bl, on="run_id", how="left")
                .with_columns((pl.col(_col) - pl.col("_base")).alias(_col))
                .drop("_base")
            )
        # Recompute emg_mean after baseline correction
        emg_ov_run_avg = emg_ov_run_avg.with_columns(
            pl.mean_horizontal(emg_ov_cols).alias("emg_mean")
        )

    emg_ov_avg = (
        emg_ov_run_avg.group_by("time_sec", "is_robot")
        .agg(pl.col("emg_mean").mean().alias("emg"))
        .sort("time_sec")
    )

    emg_ov_robot = emg_ov_avg.filter(pl.col("is_robot") == True)
    emg_ov_no_robot = emg_ov_avg.filter(pl.col("is_robot") == False)

    emg_ov_fig, (emg_ov_ax1, emg_ov_ax2) = plt.subplots(1, 2, figsize=(14, 5))

    emg_ov_ax1.plot(
        emg_ov_robot["time_sec"], emg_ov_robot["emg"], color="tab:blue"
    )
    emg_ov_ax1.set_title("Robot Trials")
    emg_ov_ax1.set_xlabel("Time (s)")
    emg_ov_ax1.set_ylabel("EMG (mV)")
    emg_ov_ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    emg_ov_ax2.plot(
        emg_ov_no_robot["time_sec"], emg_ov_no_robot["emg"], color="tab:orange"
    )
    emg_ov_ax2.set_title("No-Robot Trials")
    emg_ov_ax2.set_xlabel("Time (s)")
    emg_ov_ax2.set_ylabel("EMG (mV)")
    emg_ov_ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(df, mo, pl):
    # EMG Overall Summary Stats
    # Reuses: df, pl, mo from existing cells

    emg_ov_statsFiltered = df.filter(
        (pl.col("time_sec") >= -5.0) & (pl.col("time_sec") <= 15.0)
    )
    emg_ov_emg_cols = [c for c in df.columns if "EMG" in c and c.endswith("(mV)")]
    emg_ov_bl = emg_ov_statsFiltered.filter(pl.col("time_sec") < 0)

    emg_ov_lines = []
    for _rob, _cond in [(True, "Robot"), (False, "No-Robot")]:
        _cdf = emg_ov_statsFiltered.filter(pl.col("is_robot") == _rob)
        _runs = _cdf.select("run_id", "task_instance").unique().height
        _parts = _cdf["participant"].n_unique()
        _tasks = _cdf["task"].n_unique()
        _ra = _cdf.with_columns(pl.mean_horizontal(emg_ov_emg_cols).alias("_emg"))
        _emg_m = _ra["_emg"].mean()
        _emg_x = _ra["_emg"].max()
        # Baseline
        _bl_cdf = emg_ov_bl.filter(pl.col("is_robot") == _rob)
        _bl_ra = _bl_cdf.with_columns(
            pl.mean_horizontal(emg_ov_emg_cols).alias("_emg")
        )
        _emg_bl = _bl_ra["_emg"].mean() if _bl_ra.height > 0 else 0.0
        emg_ov_lines.append(
            f"| {_cond} | {_runs} | {_parts} | {_tasks} | {_emg_m:.4f} | {_emg_x:.4f} | {_emg_bl:.4f} |"
        )

    emg_ov_tbl = "| Condition | Epochs | Participants | Tasks | EMG Mean (mV) | EMG Max (mV) | EMG Baseline (mV) |\n"
    emg_ov_tbl += "|-----------|--------|--------------|-------|---------------|--------------|-------------------|\n"
    emg_ov_tbl += "\n".join(emg_ov_lines)

    mo.md(f"""
    ### EMG Overall Summary

    Global sensor average across all tasks, participants, and time points (-5 to 15s).

    {emg_ov_tbl}
    """)
    return


@app.cell(hide_code=True)
def _(df, mo):
    # EMG Sensor Selector Widget
    # Reuses: mo, df from existing cells

    emg_sensor_options = {}
    for _c in df.columns:
        if "EMG" in _c and _c.endswith("(mV)"):
            # Clean label: "Avanti 1 (82703)" or "Duo 3 EMG1 (78042)"
            _parts = _c.split(" | ")
            _sensor_full = _parts[0]  # "Avanti Sensor 1 (82703)"
            _emg_ch = _parts[1].replace(" (mV)", "")  # "EMG 1" or "EMG 2"
            # Extract short sensor name
            _words = _sensor_full.split()
            _short = f"{_words[0]} {_words[2]} ({_words[3].strip('()')})"
            if "Duo" in _sensor_full:
                _label = f"{_short} {_emg_ch}"
            else:
                _label = _short
            emg_sensor_options[_label] = _c

    emg_sensor_selector = mo.ui.multiselect(
        options=list(emg_sensor_options.keys()),
        value=list(emg_sensor_options.keys()),
        label="Select EMG sensors",
    )
    emg_sensor_selector
    return emg_sensor_options, emg_sensor_selector


@app.cell(hide_code=True)
def _(df, emg_sensor_options, emg_sensor_selector, mo, pl):
    # EMG Sensor Plot Summary Stats
    # Reuses: df, pl, mo, emg_sensor_selector, emg_sensor_options from existing cells

    emg_sens_statsFiltered = df.filter(
        (pl.col("time_sec") >= -5.0) & (pl.col("time_sec") <= 15.0)
    )
    emg_sens_bl = emg_sens_statsFiltered.filter(pl.col("time_sec") < 0)

    emg_sens_lines = []
    for _lbl in emg_sensor_selector.value:
        _col = emg_sensor_options[_lbl]
        for _rob, _cond in [(True, "Robot"), (False, "No-Robot")]:
            _vals = (
                emg_sens_statsFiltered.filter(pl.col("is_robot") == _rob)[_col]
                .drop_nulls()
                .to_numpy()
            )
            if len(_vals) == 0:
                continue
            _bl_vals = (
                emg_sens_bl.filter(pl.col("is_robot") == _rob)[_col]
                .drop_nulls()
                .to_numpy()
            )
            _bl_mean = _bl_vals.mean() if len(_bl_vals) > 0 else 0.0
            emg_sens_lines.append(
                f"| {_lbl} | {_cond} | {_vals.mean():.4f} | {_vals.max():.4f} | {_bl_mean:.4f} |"
            )

    emg_sens_tbl = (
        "| Sensor | Condition | Mean (mV) | Max (mV) | Baseline (mV) |\n"
    )
    emg_sens_tbl += (
        "|--------|-----------|-----------|----------|---------------|\n"
    )
    emg_sens_tbl += "\n".join(emg_sens_lines)

    mo.md(f"""
    ### EMG Per-Sensor Summary

    Mean and max EMG (mV) per sensor, across all runs and time points.

    {emg_sens_tbl}
    """)
    return


@app.cell(hide_code=True)
def _(
    Line2D,
    df,
    emg_baseline_switch,
    emg_sensor_options,
    emg_sensor_selector,
    pl,
    plt,
):
    # EMG Per-Sensor Interactive Plot
    # Reuses: df, pl, plt, np, Line2D from existing cells
    import matplotlib.gridspec as gridspec

    emg_sens_TIME_MIN = -5.0
    emg_sens_TIME_MAX = 15.0

    emg_sens_filtered = df.filter(
        (pl.col("time_sec") >= emg_sens_TIME_MIN)
        & (pl.col("time_sec") <= emg_sens_TIME_MAX)
    )

    emg_sens_selected = [
        emg_sensor_options[_ch] for _ch in emg_sensor_selector.value
    ]

    # Baseline correction
    if emg_baseline_switch.value:
        for _col in emg_sens_selected:
            _bl = (
                emg_sens_filtered.filter(pl.col("time_sec") < 0)
                .group_by("run_id")
                .agg(pl.col(_col).mean().alias("_base"))
            )
            emg_sens_filtered = (
                emg_sens_filtered.join(_bl, on="run_id", how="left")
                .with_columns((pl.col(_col) - pl.col("_base")).alias(_col))
                .drop("_base")
            )

    if not emg_sens_selected:
        emg_sens_fig, _ax = plt.subplots(figsize=(14, 5))
        _ax.text(
            0.5, 0.5, "No sensors selected", ha="center", va="center", fontsize=14
        )
        _ax.set_axis_off()
        plt.show()
    else:
        emg_sens_time = None
        emg_sens_robot = {}
        emg_sens_no_robot = {}

        for _col in emg_sens_selected:
            _avg = (
                emg_sens_filtered.group_by("time_sec", "is_robot")
                .agg(pl.col(_col).mean().alias("val"))
                .sort("time_sec")
            )

            if emg_sens_time is None:
                emg_sens_time = _avg.filter(pl.col("is_robot") == True)[
                    "time_sec"
                ].to_numpy()

            emg_sens_robot[_col] = _avg.filter(pl.col("is_robot") == True)[
                "val"
            ].to_numpy()
            emg_sens_no_robot[_col] = _avg.filter(pl.col("is_robot") == False)[
                "val"
            ].to_numpy()

        # One unique color per sensor
        _n_sens = len(emg_sens_selected)
        _cmap = plt.cm.tab10 if _n_sens <= 10 else plt.cm.tab20
        emg_sens_colors = {
            _c: _cmap(_i / max(_n_sens - 1, 1))
            for _i, _c in enumerate(emg_sens_selected)
        }

        emg_sens_fig, (emg_sens_ax1, emg_sens_ax2) = plt.subplots(
            1, 2, figsize=(14, 5)
        )

        for _lbl in emg_sensor_selector.value:
            _col = emg_sensor_options[_lbl]
            _c = emg_sens_colors[_col]
            emg_sens_ax1.plot(
                emg_sens_time, emg_sens_robot[_col], color=_c, label=_lbl
            )
            emg_sens_ax2.plot(
                emg_sens_time, emg_sens_no_robot[_col], color=_c, label=_lbl
            )

        emg_sens_ax1.set_title("Robot Trials")
        emg_sens_ax1.set_xlabel("Time (s)")
        emg_sens_ax1.set_ylabel("EMG (mV)")
        emg_sens_ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        emg_sens_ax2.set_title("No-Robot Trials")
        emg_sens_ax2.set_xlabel("Time (s)")
        emg_sens_ax2.set_ylabel("EMG (mV)")
        emg_sens_ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        # Legend handles
        _legend_handles = []
        for _lbl in emg_sensor_selector.value:
            _col = emg_sensor_options[_lbl]
            _c = emg_sens_colors[_col]
            _legend_handles.append(Line2D([0], [0], color=_c, label=_lbl))

        _n_actual = len(_legend_handles)
        _ncols_actual = min(_n_actual, 8)
        _n_legend_rows = (_n_actual + _ncols_actual - 1) // _ncols_actual

        emg_sens_fig.set_size_inches(14, 5 + 0.35 * _n_legend_rows)
        _bottom = min(0.34, 0.12 + 0.045 * _n_legend_rows)

        emg_sens_fig.legend(
            handles=_legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=_ncols_actual,
            fontsize=9,
            frameon=False,
            handlelength=2,
            handletextpad=0.5,
            columnspacing=1.5,
        )

        emg_sens_fig.subplots_adjust(
            left=0.06,
            right=0.99,
            top=0.93,
            bottom=_bottom,
            wspace=0.20,
        )

        plt.show()
    return


@app.cell(hide_code=True)
def _(df, mo):
    # EMG Task Selector Widget
    # Reuses: mo, df from existing cells

    _emg_task_sorted = sorted(
        df["task"].unique().to_list(), key=lambda _x: int(_x.split("_")[1])
    )

    emg_task_selector = mo.ui.multiselect(
        options=_emg_task_sorted, value=_emg_task_sorted, label="Select tasks"
    )
    emg_task_selector
    return (emg_task_selector,)


@app.cell(hide_code=True)
def _(df, emg_task_selector, mo, pl):
    # EMG Task Plot Summary Stats
    # Reuses: df, pl, mo, emg_task_selector from existing cells

    emg_task_statsFiltered = df.filter(
        (pl.col("time_sec") >= -5.0) & (pl.col("time_sec") <= 15.0)
    )
    emg_task_emg_cols_s = [
        c for c in df.columns if "EMG" in c and c.endswith("(mV)")
    ]
    emg_task_bl = emg_task_statsFiltered.filter(pl.col("time_sec") < 0)

    emg_task_lines = []
    for _task in emg_task_selector.value:
        _tdf = emg_task_statsFiltered.filter(pl.col("task") == _task)
        _bl_tdf = emg_task_bl.filter(pl.col("task") == _task)
        for _rob, _lbl in [(True, "Robot"), (False, "No-Robot")]:
            _cdf = _tdf.filter(pl.col("is_robot") == _rob)
            _epochs = _cdf.select("run_id", "task_instance").unique().height
            if _epochs == 0:
                continue
            _ra = _cdf.with_columns(
                pl.mean_horizontal(emg_task_emg_cols_s).alias("_emg")
            )
            _emg_m = _ra["_emg"].mean()
            _emg_x = _ra["_emg"].max()
            # Baseline
            _bl_cdf = _bl_tdf.filter(pl.col("is_robot") == _rob)
            _bl_ra = _bl_cdf.with_columns(
                pl.mean_horizontal(emg_task_emg_cols_s).alias("_emg")
            )
            _emg_bl = _bl_ra["_emg"].mean() if _bl_ra.height > 0 else 0.0
            emg_task_lines.append(
                f"| {_task} | {_lbl} | {_epochs} | {_emg_m:.4f} | {_emg_x:.4f} | {_emg_bl:.4f} |"
            )

    emg_task_tbl = "| Task | Condition | Epochs | EMG Mean (mV) | EMG Max (mV) | EMG Baseline (mV) |\n"
    emg_task_tbl += "|------|-----------|--------|---------------|--------------|-------------------|\n"
    emg_task_tbl += "\n".join(emg_task_lines)

    mo.md(f"""
    ### EMG Per-Task Summary

    Mean and max EMG (mV) of the global sensor average, across all runs per task and condition.

    {emg_task_tbl}
    """)
    return


@app.cell(hide_code=True)
def _(Line2D, df, emg_baseline_switch, emg_task_selector, pl, plt):
    # EMG Per-Task Interactive Plot
    # Reuses: df, pl, plt, np, Line2D from existing cells

    emg_task_TIME_MIN = -5.0
    emg_task_TIME_MAX = 15.0

    emg_task_filtered = df.filter(
        (pl.col("time_sec") >= emg_task_TIME_MIN)
        & (pl.col("time_sec") <= emg_task_TIME_MAX)
    )

    emg_task_emg_cols = [
        c for c in df.columns if "EMG" in c and c.endswith("(mV)")
    ]

    emg_task_selected = emg_task_selector.value

    if not emg_task_selected:
        emg_task_fig, _ax = plt.subplots(figsize=(14, 5))
        _ax.text(
            0.5, 0.5, "No tasks selected", ha="center", va="center", fontsize=14
        )
        _ax.set_axis_off()
        plt.show()
    else:
        emg_task_time = None
        emg_task_robot = {}
        emg_task_no_robot = {}

        for _task in emg_task_selected:
            _task_df = emg_task_filtered.filter(pl.col("task") == _task)

            _task_run_avg = _task_df.with_columns(
                pl.mean_horizontal(emg_task_emg_cols).alias("emg_mean")
            )

            # Baseline correction
            if emg_baseline_switch.value:
                _bl = (
                    _task_run_avg.filter(pl.col("time_sec") < 0)
                    .group_by("run_id")
                    .agg(pl.col("emg_mean").mean().alias("_base"))
                )
                _task_run_avg = (
                    _task_run_avg.join(_bl, on="run_id", how="left")
                    .with_columns(
                        (pl.col("emg_mean") - pl.col("_base")).alias("emg_mean")
                    )
                    .drop("_base")
                )

            _task_avg = (
                _task_run_avg.group_by("time_sec", "is_robot")
                .agg(pl.col("emg_mean").mean().alias("emg"))
                .sort("time_sec")
            )

            if emg_task_time is None:
                emg_task_time = _task_avg.filter(pl.col("is_robot") == True)[
                    "time_sec"
                ].to_numpy()

            emg_task_robot[_task] = _task_avg.filter(pl.col("is_robot") == True)[
                "emg"
            ].to_numpy()
            emg_task_no_robot[_task] = _task_avg.filter(
                pl.col("is_robot") == False
            )["emg"].to_numpy()

        _n_tasks = len(emg_task_selected)
        _cmap = plt.cm.tab10 if _n_tasks <= 10 else plt.cm.tab20
        emg_task_colors = {
            _t: _cmap(_i / max(_n_tasks - 1, 1))
            for _i, _t in enumerate(emg_task_selected)
        }

        emg_task_fig, (emg_task_ax1, emg_task_ax2) = plt.subplots(
            1, 2, figsize=(14, 5)
        )

        for _task in emg_task_selected:
            _c = emg_task_colors[_task]
            emg_task_ax1.plot(
                emg_task_time, emg_task_robot[_task], color=_c, label=_task
            )
            emg_task_ax2.plot(
                emg_task_time, emg_task_no_robot[_task], color=_c, label=_task
            )

        emg_task_ax1.set_title("Robot Trials")
        emg_task_ax1.set_xlabel("Time (s)")
        emg_task_ax1.set_ylabel("EMG (mV)")
        emg_task_ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        emg_task_ax2.set_title("No-Robot Trials")
        emg_task_ax2.set_xlabel("Time (s)")
        emg_task_ax2.set_ylabel("EMG (mV)")
        emg_task_ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        # Legend handles
        _legend_handles = []
        for _task in emg_task_selected:
            _c = emg_task_colors[_task]
            _legend_handles.append(Line2D([0], [0], color=_c, label=_task))

        _n_actual = len(_legend_handles)
        _ncols_actual = min(_n_actual, 8)
        _n_legend_rows = (_n_actual + _ncols_actual - 1) // _ncols_actual

        emg_task_fig.set_size_inches(14, 5 + 0.35 * _n_legend_rows)
        _bottom = min(0.34, 0.12 + 0.045 * _n_legend_rows)

        emg_task_fig.legend(
            handles=_legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=_ncols_actual,
            fontsize=9,
            frameon=False,
            handlelength=2,
            handletextpad=0.5,
            columnspacing=1.5,
        )

        emg_task_fig.subplots_adjust(
            left=0.06,
            right=0.99,
            top=0.93,
            bottom=_bottom,
            wspace=0.20,
        )

        plt.show()
    return


if __name__ == "__main__":
    app.run()
