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
def _(df, pl, plt):
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


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(
    df,
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
def _(Line2D, df, fnirs_task_selector, pl, plt, savgol_filter):
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


if __name__ == "__main__":
    app.run()
