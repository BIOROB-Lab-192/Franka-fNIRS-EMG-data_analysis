import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")

# ═══════════════════════════════════════════════════════════════
# graphing.py — Multi-Sensor Visualization Dashboard
# ═══════════════════════════════════════════════════════════════
# Interactive marimo notebook for visualizing fNIRS and EMG data
# from the combined data packet. Sections:
#   1. Data loading
#   2. Plot helpers (baseline correction, legend layout)
#   3. fNIRS — aggregated, per-channel, per-task plots + stats
#   4. EMG — aggregated, per-sensor, per-task plots + stats
#   5. Per-run viewer — stacked fNIRS + EMG for individual epochs
#
# All plots support optional baseline correction (-5 to 0s).
# Data loaded from data/processed/combined/data_packet/.
# ═══════════════════════════════════════════════════════════════


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import numpy as np

    return mo, np, pl, plt


@app.cell
def _(mo, pl):
    DATA_BASE = "./data/processed/combined/data_packet"

    fnirs_df = pl.read_parquet(f"{DATA_BASE}/fnirs_full.parquet")
    emg_df = pl.read_parquet(f"{DATA_BASE}/emg_full.parquet")
    robot_df = pl.read_parquet(f"{DATA_BASE}/robot_full.parquet")
    epoch_index = pl.read_csv(f"{DATA_BASE}/epoch_index.csv")

    # Add 'task' column to emg_df (it only has task_instance, not task)
    emg_df = emg_df.join(
        epoch_index.select("run_id", "task_instance", "task_id"),
        on=["run_id", "task_instance"],
        how="left",
    ).rename({"task_id": "task"})

    # Add 'task' column to robot_df (it has task_id, rename to task)
    robot_df = robot_df.rename({"task_id": "task"})

    mo.md(f"""
    **Data loaded:**
    - fNIRS: `{fnirs_df.shape[0]:,}` rows × {fnirs_df.shape[1]} cols (10 Hz)
    - EMG: `{emg_df.shape[0]:,}` rows × {emg_df.shape[1]} cols (~1926 Hz)
    - Robot: `{robot_df.shape[0]:,}` rows × {robot_df.shape[1]} cols
    - Epoch index: {epoch_index.shape[0]} epochs
    """)
    return emg_df, fnirs_df


@app.cell
def _(np):
    import json
    import ast


    def unflattern_HT(flat):
        flat = ast.literal_eval(flat)
        T = np.array(flat).reshape(4, 4)
        T = T.T
        return T

    return


@app.cell(hide_code=True)
def _(pl):
    # Reusable helpers for all plots
    # Reuses: pl, np from imports cell
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D


    def apply_baseline(df, signal_cols, time_col="time_sec", baseline_end=0):
        """Subtract per-run baseline (mean of time < baseline_end) from signal columns.
        Returns a new DataFrame with baseline-corrected columns."""
        _bl = (
            df.filter(pl.col(time_col) < baseline_end)
            .group_by("run_id")
            .agg(
                [
                    pl.col(c).drop_nulls().mean().alias(f"{c}_bl")
                    for c in signal_cols
                ]
            )
        )
        _result = df.join(_bl, on="run_id", how="left")
        for c in signal_cols:
            _result = _result.with_columns(
                (pl.col(c) - pl.col(f"{c}_bl")).alias(c)
            ).drop(f"{c}_bl")
        return _result


    def legend_layout(fig, n_handles, ncols=8):
        """Apply standard legend-below layout with dynamic sizing."""
        _n_rows = (n_handles + ncols - 1) // ncols
        fig.set_size_inches(14, 5 + 0.35 * _n_rows)
        _bottom = min(0.34, 0.12 + 0.045 * _n_rows)
        fig.subplots_adjust(
            left=0.06, right=0.99, top=0.93, bottom=_bottom, wspace=0.20
        )


    def build_legend(fig, handles, ncols=8, fontsize=9):
        """Place legend below figure, centered."""
        _nc = min(len(handles), ncols)
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=_nc,
            fontsize=fontsize,
            frameon=False,
            handlelength=2,
            handletextpad=0.5,
            columnspacing=1.5,
        )

    return Line2D, apply_baseline, build_legend, legend_layout


@app.cell(hide_code=True)
def fnirs_baseline_switch(mo):
    # ─── fNIRS Plots ───────────────────────────────────────────
    fnirs_baseline_switch = mo.ui.switch(label="Baseline correction (-5 to 0s)")
    fnirs_baseline_switch
    return (fnirs_baseline_switch,)


@app.cell(hide_code=True)
def _(fnirs_baseline_switch, fnirs_df, pl, plt):
    # fNIRS Aggregated: Robot vs No-Robot (Global Channel Average)
    # Reuses: fnirs_df, pl, plt, np from existing cells

    fnirs_TIME_MIN = -5.0
    fnirs_TIME_MAX = 15.0
    fnirs_UM_CONVERSION = 1e6

    # Filter to time window
    fnirs_filtered = fnirs_df.filter(
        (pl.col("time_sec") >= fnirs_TIME_MIN)
        & (pl.col("time_sec") <= fnirs_TIME_MAX)
    )

    # Identify fNIRS channels
    fnirs_hbo_cols = [c for c in fnirs_df.columns if c.endswith("_hbo")]
    fnirs_hbr_cols = [c for c in fnirs_df.columns if c.endswith("_hbr")]

    # Average channels within each run first
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

    # Plot
    fnirs_fig, (fnirs_ax1, fnirs_ax2) = plt.subplots(1, 2, figsize=(14, 5))

    fnirs_ax1.plot(
        fnirs_robot["time_sec"], fnirs_robot["hbo"], label="HbO", color="red"
    )
    fnirs_ax1.plot(
        fnirs_robot["time_sec"], fnirs_robot["hbr"], label="HbR", color="blue"
    )
    fnirs_ax1.set_title("Robot Trials")
    fnirs_ax1.set_xlabel("Time (s)")
    fnirs_ax1.set_ylabel("Concentration (μM)")
    fnirs_ax1.legend()
    fnirs_ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    fnirs_ax2.plot(
        fnirs_no_robot["time_sec"], fnirs_no_robot["hbo"], label="HbO", color="red"
    )
    fnirs_ax2.plot(
        fnirs_no_robot["time_sec"],
        fnirs_no_robot["hbr"],
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
    return


@app.cell(hide_code=True)
def _(fnirs_df, mo, pl):
    # fNIRS Overall Summary Stats
    # Reuses: fnirs_df, pl, mo from existing cells

    _fnirs_ov_filtered = fnirs_df.filter(
        (pl.col("time_sec") >= -5.0) & (pl.col("time_sec") <= 15.0)
    )
    _fnirs_ov_hbo = [c for c in fnirs_df.columns if c.endswith("_hbo")]
    _fnirs_ov_hbr = [c for c in fnirs_df.columns if c.endswith("_hbr")]

    # Baseline data (time_sec < 0)
    _fnirs_ov_bl = _fnirs_ov_filtered.filter(pl.col("time_sec") < 0)

    _fnirs_ov_lines = []
    for _rob, _cond in [(True, "Robot"), (False, "No-Robot")]:
        _cdf = _fnirs_ov_filtered.filter(pl.col("is_robot") == _rob)
        _runs = _cdf["run_id"].n_unique()
        _parts = _cdf["participant"].n_unique()
        _tasks = _cdf["task"].n_unique()
        _cdf_post = _cdf.filter(pl.col("time_sec") >= 0)
        _ra = _cdf_post.with_columns(
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

    Global channel average across all tasks, participants. Mean and max over 0\u201315s post-stimulus; baseline from \u22125 to 0s.

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
def _(fnirs_channel_options, fnirs_channel_selector, fnirs_df, mo, pl):
    # fNIRS Channel Plot Summary Stats
    # Reuses: fnirs_df, pl, mo, fnirs_channel_selector, fnirs_channel_options from existing cells

    _fnirs_ch_statsFiltered = fnirs_df.filter(
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
                _fnirs_ch_statsFiltered.filter(
                    (pl.col("is_robot") == _rob) & (pl.col("time_sec") >= 0)
                )[_col].to_numpy()
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


@app.cell(hide_code=True)
def _(
    Line2D,
    build_legend,
    fnirs_baseline_switch,
    fnirs_channel_options,
    fnirs_channel_selector,
    fnirs_df,
    legend_layout,
    pl,
    plt,
):
    # fNIRS Per-Channel Interactive Plot
    # Reuses: fnirs_df, pl, plt, np from existing cells

    fnirs_ch_TIME_MIN = -5.0
    fnirs_ch_TIME_MAX = 15.0
    fnirs_ch_UM = 1e6

    # Filter to time window
    fnirs_ch_filtered = fnirs_df.filter(
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

    # Build pair -> {HbO: col, HbR: col}
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

            fnirs_ch_robot[_col] = _avg.filter(pl.col("is_robot") == True)[
                "val"
            ].to_numpy()
            fnirs_ch_no_robot[_col] = _avg.filter(pl.col("is_robot") == False)[
                "val"
            ].to_numpy()

        # Colors per pair
        _pair_names = sorted(fnirs_ch_pairs.keys())
        _cmap = plt.cm.tab10 if len(_pair_names) <= 10 else plt.cm.tab20
        fnirs_ch_pair_colors = {
            _p: _cmap(_i / max(len(_pair_names) - 1, 1))
            for _i, _p in enumerate(_pair_names)
        }

        fnirs_ch_fig, (fnirs_ch_ax1, fnirs_ch_ax2) = plt.subplots(
            1, 2, figsize=(14, 5)
        )

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

        legend_layout(fnirs_ch_fig, len(_legend_handles))
        build_legend(fnirs_ch_fig, _legend_handles)

        plt.show()

    fnirs_ch_fig
    return


@app.cell(hide_code=True)
def _(fnirs_df, mo):
    # fNIRS Task Selector Widget
    # Reuses: mo, df from existing cells

    _fnirs_task_sorted = sorted(
        fnirs_df["task"].unique().to_list(), key=lambda _x: int(_x.split("_")[1])
    )

    fnirs_task_selector = mo.ui.multiselect(
        options=_fnirs_task_sorted, value=_fnirs_task_sorted, label="Select tasks"
    )
    fnirs_task_selector
    return (fnirs_task_selector,)


@app.cell(hide_code=True)
def _(fnirs_df, fnirs_task_selector, mo, pl):
    # fNIRS Task Plot Summary Stats
    # Reuses: fnirs_df, pl, mo, fnirs_task_selector from existing cells

    _fnirs_task_statsFiltered = fnirs_df.filter(
        (pl.col("time_sec") >= -5.0) & (pl.col("time_sec") <= 15.0)
    )
    _fnirs_task_hbo_cols_s = [c for c in fnirs_df.columns if c.endswith("_hbo")]
    _fnirs_task_hbr_cols_s = [c for c in fnirs_df.columns if c.endswith("_hbr")]
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
            _cdf_post = _cdf.filter(pl.col("time_sec") >= 0)
            _ra = _cdf_post.with_columns(
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
    apply_baseline,
    build_legend,
    fnirs_baseline_switch,
    fnirs_df,
    fnirs_task_selector,
    legend_layout,
    pl,
    plt,
):
    # fNIRS Per-Task Interactive Plot
    # Reuses: fnirs_df, pl, plt, np, Line2D from existing cells

    fnirs_task_TIME_MIN = -5.0
    fnirs_task_TIME_MAX = 15.0
    fnirs_task_UM = 1e6

    # Filter to time window
    fnirs_task_filtered = fnirs_df.filter(
        (pl.col("time_sec") >= fnirs_task_TIME_MIN)
        & (pl.col("time_sec") <= fnirs_task_TIME_MAX)
    )

    # Identify fNIRS channels
    fnirs_task_hbo_cols = [c for c in fnirs_df.columns if c.endswith("_hbo")]
    fnirs_task_hbr_cols = [c for c in fnirs_df.columns if c.endswith("_hbr")]

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
        fnirs_task_time = None
        fnirs_task_robot = {}
        fnirs_task_no_robot = {}

        for _task in fnirs_task_selected:
            _task_df = fnirs_task_filtered.filter(pl.col("task") == _task)

            _task_run_avg = _task_df.with_columns(
                [
                    pl.mean_horizontal(fnirs_task_hbo_cols).alias("hbo_mean"),
                    pl.mean_horizontal(fnirs_task_hbr_cols).alias("hbr_mean"),
                ]
            )

            if fnirs_baseline_switch.value:
                _task_run_avg = apply_baseline(
                    _task_run_avg, ["hbo_mean", "hbr_mean"]
                )

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

            fnirs_task_robot[_task] = {
                "hbo": _task_avg.filter(pl.col("is_robot") == True)[
                    "hbo"
                ].to_numpy(),
                "hbr": _task_avg.filter(pl.col("is_robot") == True)[
                    "hbr"
                ].to_numpy(),
            }
            fnirs_task_no_robot[_task] = {
                "hbo": _task_avg.filter(pl.col("is_robot") == False)[
                    "hbo"
                ].to_numpy(),
                "hbr": _task_avg.filter(pl.col("is_robot") == False)[
                    "hbr"
                ].to_numpy(),
            }

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

        _legend_handles = []
        for _task in fnirs_task_selected:
            _c = fnirs_task_colors[_task]
            _legend_handles.append(
                Line2D([0], [0], color=_c, linestyle="-", label=f"{_task} HbO")
            )
            _legend_handles.append(
                Line2D([0], [0], color=_c, linestyle="--", label=f"{_task} HbR")
            )

        legend_layout(fnirs_task_fig, len(_legend_handles))
        build_legend(fnirs_task_fig, _legend_handles)

        plt.show()

    fnirs_task_fig
    return


@app.cell(hide_code=True)
def emg_baseline_switch(mo):
    # ─── EMG Plots ─────────────────────────────────────────────
    emg_baseline_switch = mo.ui.switch(label="Baseline correction (-5 to 0s)")
    emg_baseline_switch
    return (emg_baseline_switch,)


@app.cell(hide_code=True)
def _(emg_baseline_switch, emg_df, pl, plt):
    # EMG Aggregated: Robot vs No-Robot (Global Sensor Average)
    # Reuses: emg_df, pl, plt, np from existing cells

    emg_ov_TIME_MIN = -5.0
    emg_ov_TIME_MAX = 15.0

    emg_ov_cols = [c for c in emg_df.columns if "EMG" in c and c.endswith("(mV)")]

    emg_ov_filtered = emg_df.filter(
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

    emg_ov_fig
    return


@app.cell(hide_code=True)
def _(emg_df, mo, pl):
    # EMG Overall Summary Stats
    # Reuses: emg_df, pl, mo from existing cells

    emg_ov_statsFiltered = emg_df.filter(
        (pl.col("time_sec") >= -5.0) & (pl.col("time_sec") <= 15.0)
    )
    emg_ov_emg_cols = [
        c for c in emg_df.columns if "EMG" in c and c.endswith("(mV)")
    ]
    emg_ov_bl = emg_ov_statsFiltered.filter(pl.col("time_sec") < 0)

    emg_ov_lines = []
    for _rob, _cond in [(True, "Robot"), (False, "No-Robot")]:
        _cdf = emg_ov_statsFiltered.filter(pl.col("is_robot") == _rob)
        _runs = _cdf.select("run_id", "task_instance").unique().height
        _parts = _cdf["participant"].n_unique()
        _tasks = _cdf["task"].n_unique()
        _cdf_post = _cdf.filter(pl.col("time_sec") >= 0)
        _ra = _cdf_post.with_columns(
            pl.mean_horizontal(emg_ov_emg_cols).alias("_emg")
        )
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

    Global sensor average across all tasks, participants. Mean and max over 0\u201315s post-stimulus; baseline from \u22125 to 0s.

    {emg_ov_tbl}
    """)
    return


@app.cell(hide_code=True)
def _(emg_df, mo):
    # EMG Sensor Selector Widget
    # Reuses: mo, df from existing cells

    emg_sensor_options = {}
    for _c in emg_df.columns:
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
def _(emg_df, emg_sensor_options, emg_sensor_selector, mo, pl):
    # EMG Sensor Plot Summary Stats
    # Reuses: emg_df, pl, mo, emg_sensor_selector, emg_sensor_options from existing cells

    emg_sens_statsFiltered = emg_df.filter(
        (pl.col("time_sec") >= -5.0) & (pl.col("time_sec") <= 15.0)
    )
    emg_sens_bl = emg_sens_statsFiltered.filter(pl.col("time_sec") < 0)

    emg_sens_lines = []
    for _lbl in emg_sensor_selector.value:
        _col = emg_sensor_options[_lbl]
        for _rob, _cond in [(True, "Robot"), (False, "No-Robot")]:
            _vals = (
                emg_sens_statsFiltered.filter(
                    (pl.col("is_robot") == _rob) & (pl.col("time_sec") >= 0)
                )[_col]
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

    Mean and max EMG (mV) per sensor, across all runs (0\u201315s post-stimulus).

    {emg_sens_tbl}
    """)
    return


@app.cell(hide_code=True)
def _(
    Line2D,
    build_legend,
    emg_baseline_switch,
    emg_df,
    emg_sensor_options,
    emg_sensor_selector,
    legend_layout,
    pl,
    plt,
):
    # EMG Per-Sensor Interactive Plot
    # Reuses: emg_df, pl, plt, np, Line2D from existing cells

    emg_sens_TIME_MIN = -5.0
    emg_sens_TIME_MAX = 15.0

    emg_sens_filtered = emg_df.filter(
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
        emg_sens_robot = {}
        emg_sens_no_robot = {}

        for _col in emg_sens_selected:
            _avg = (
                emg_sens_filtered.group_by("time_sec", "is_robot")
                .agg(pl.col(_col).mean().alias("val"))
                .sort("time_sec")
            )
            emg_sens_robot[_col] = _avg.filter(pl.col("is_robot") == True)
            emg_sens_no_robot[_col] = _avg.filter(pl.col("is_robot") == False)

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
                emg_sens_robot[_col]["time_sec"].to_numpy(),
                emg_sens_robot[_col]["val"].to_numpy(),
                color=_c,
                label=_lbl,
                linewidth=0.5,
            )
            emg_sens_ax2.plot(
                emg_sens_no_robot[_col]["time_sec"].to_numpy(),
                emg_sens_no_robot[_col]["val"].to_numpy(),
                color=_c,
                label=_lbl,
                linewidth=0.5,
            )

        emg_sens_ax1.set_title("Robot Trials")
        emg_sens_ax1.set_xlabel("Time (s)")
        emg_sens_ax1.set_ylabel("EMG (mV)")
        emg_sens_ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        emg_sens_ax2.set_title("No-Robot Trials")
        emg_sens_ax2.set_xlabel("Time (s)")
        emg_sens_ax2.set_ylabel("EMG (mV)")
        emg_sens_ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        # Legend
        _legend_handles = []
        for _lbl in emg_sensor_selector.value:
            _col = emg_sensor_options[_lbl]
            _c = emg_sens_colors[_col]
            _legend_handles.append(Line2D([0], [0], color=_c, label=_lbl))

        legend_layout(emg_sens_fig, len(_legend_handles))
        build_legend(emg_sens_fig, _legend_handles)

        plt.show()

    emg_sens_fig
    return


@app.cell(hide_code=True)
def _(emg_df, mo):
    # EMG Task Selector Widget
    # Reuses: mo, df from existing cells

    _emg_task_sorted = sorted(
        emg_df["task"].unique().to_list(), key=lambda _x: int(_x.split("_")[1])
    )

    emg_task_selector = mo.ui.multiselect(
        options=_emg_task_sorted, value=_emg_task_sorted, label="Select tasks"
    )
    emg_task_selector
    return (emg_task_selector,)


@app.cell(hide_code=True)
def _(emg_df, emg_task_selector, mo, pl):
    # EMG Task Plot Summary Stats
    # Reuses: emg_df, pl, mo, emg_task_selector from existing cells

    emg_task_statsFiltered = emg_df.filter(
        (pl.col("time_sec") >= -5.0) & (pl.col("time_sec") <= 15.0)
    )
    emg_task_emg_cols_s = [
        c for c in emg_df.columns if "EMG" in c and c.endswith("(mV)")
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
            _cdf_post = _cdf.filter(pl.col("time_sec") >= 0)
            _ra = _cdf_post.with_columns(
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

    Mean and max EMG (mV) of the global sensor average, across all runs per task and condition (0\u201315s post-stimulus).

    {emg_task_tbl}
    """)
    return


@app.cell(hide_code=True)
def _(
    Line2D,
    apply_baseline,
    build_legend,
    emg_baseline_switch,
    emg_df,
    emg_task_selector,
    legend_layout,
    pl,
    plt,
):
    # EMG Per-Task Interactive Plot
    # Reuses: emg_df, pl, plt, np, Line2D from existing cells

    emg_task_TIME_MIN = -5.0
    emg_task_TIME_MAX = 15.0

    emg_task_filtered = emg_df.filter(
        (pl.col("time_sec") >= emg_task_TIME_MIN)
        & (pl.col("time_sec") <= emg_task_TIME_MAX)
    )

    emg_task_emg_cols = [
        c for c in emg_df.columns if "EMG" in c and c.endswith("(mV)")
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
        emg_task_robot = {}
        emg_task_no_robot = {}

        for _task in emg_task_selected:
            _task_df = emg_task_filtered.filter(pl.col("task") == _task)

            _task_run_avg = _task_df.with_columns(
                pl.mean_horizontal(emg_task_emg_cols).alias("emg_mean")
            )

            # Baseline correction
            if emg_baseline_switch.value:
                _task_run_avg = apply_baseline(_task_run_avg, ["emg_mean"])

            _task_avg = (
                _task_run_avg.group_by("time_sec", "is_robot")
                .agg(pl.col("emg_mean").mean().alias("emg"))
                .sort("time_sec")
            )

            emg_task_robot[_task] = _task_avg.filter(pl.col("is_robot") == True)
            emg_task_no_robot[_task] = _task_avg.filter(
                pl.col("is_robot") == False
            )

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
                emg_task_robot[_task]["time_sec"].to_numpy(),
                emg_task_robot[_task]["emg"].to_numpy(),
                color=_c,
                label=_task,
                linewidth=0.5,
            )
            emg_task_ax2.plot(
                emg_task_no_robot[_task]["time_sec"].to_numpy(),
                emg_task_no_robot[_task]["emg"].to_numpy(),
                color=_c,
                label=_task,
                linewidth=0.5,
            )

        emg_task_ax1.set_title("Robot Trials")
        emg_task_ax1.set_xlabel("Time (s)")
        emg_task_ax1.set_ylabel("EMG (mV)")
        emg_task_ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        emg_task_ax2.set_title("No-Robot Trials")
        emg_task_ax2.set_xlabel("Time (s)")
        emg_task_ax2.set_ylabel("EMG (mV)")
        emg_task_ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        # Legend
        _legend_handles = []
        for _task in emg_task_selected:
            _c = emg_task_colors[_task]
            _legend_handles.append(Line2D([0], [0], color=_c, label=_task))

        legend_layout(emg_task_fig, len(_legend_handles))
        build_legend(emg_task_fig, _legend_handles)

        plt.show()

    emg_task_fig
    return


@app.cell(hide_code=True)
def _(fnirs_df, mo):
    # ─── Per-Run Viewer ────────────────────────────────────────
    # Per-Run Viewer Widgets
    # Reuses: mo, fnirs_df, emg_df from existing cells

    pr_task_sorted = sorted(
        fnirs_df["task"].unique().to_list(), key=lambda _x: int(_x.split("_")[1])
    )

    pr_task_select = mo.ui.dropdown(options=pr_task_sorted, label="Task")

    pr_robot_select = mo.ui.radio(
        options=["Robot", "No-Robot"],
        label="Condition",
        value="Robot",
    )

    pr_run_select = mo.ui.dropdown(
        options=sorted(fnirs_df["run_id"].unique().to_list()), label="Run"
    )

    pr_baseline_switch = mo.ui.switch(label="Baseline correction (-5 to 0s)")

    mo.md("### Per-Run Viewer")
    mo.vstack([pr_task_select, pr_robot_select, pr_run_select, pr_baseline_switch])
    return pr_baseline_switch, pr_robot_select, pr_run_select, pr_task_select


@app.cell(hide_code=True)
def _(
    emg_df,
    fnirs_df,
    mo,
    pl,
    pr_emg_cols,
    pr_emg_labels,
    pr_hbo_cols,
    pr_hbr_cols,
    pr_robot_select,
    pr_run_select,
    pr_task_select,
):
    # Per-Run Viewer Summary Stats
    # Reuses: fnirs_df, emg_df, pr_task_select, pr_robot_select, pr_run_select from existing cells

    _lines = []

    if pr_run_select.value is None or pr_task_select.value is None:
        _lines.append("*Select a run and task to see summary statistics.*")
    else:
        _cond = (
            (pl.col("run_id") == pr_run_select.value)
            & (pl.col("task") == pr_task_select.value)
            & (pl.col("is_robot") == (pr_robot_select.value == "Robot"))
        )
        _cond_post = _cond & (pl.col("time_sec") >= 0)

        _r_fnirs = fnirs_df.filter(_cond)
        _r_emg = emg_df.filter(_cond)
        _r_fnirs_post = fnirs_df.filter(_cond_post)
        _r_emg_post = emg_df.filter(_cond_post)

        _fnirs_freq = "10 Hz"
        _emg_freq = "N/A"
        if _r_emg.height > 1:
            _emg_dt = (
                _r_emg.sort("time_sec")["time_sec"].diff().drop_nulls().median()
            )
            if _emg_dt > 0:
                _emg_freq = f"{1.0 / _emg_dt:.0f} Hz"

        _participant = (
            _r_fnirs["participant"][0]
            if _r_fnirs.height > 0
            else (_r_emg["participant"][0] if _r_emg.height > 0 else "\u2014")
        )

        _lines.append(
            f"**{_participant}** \u2014 {pr_run_select.value} \u2014 {pr_task_select.value} \u2014 {pr_robot_select.value}"
        )
        _lines.append("")

        # fNIRS stats
        _fnirs_n = 0
        if _r_fnirs_post.height > 0 and len(pr_hbo_cols) > 0:
            _hbo_vals = _r_fnirs_post.select(
                pl.mean_horizontal(pr_hbo_cols)
            ).drop_nulls()
            _hbr_vals = _r_fnirs_post.select(
                pl.mean_horizontal(pr_hbr_cols)
            ).drop_nulls()
            if len(_hbo_vals) > 0:
                _hbo_mean = _hbo_vals.mean().item() * 1e6
                _hbo_peak = _hbo_vals.max().item() * 1e6
                _hbr_mean = _hbr_vals.mean().item() * 1e6
                _hbr_peak = _hbr_vals.max().item() * 1e6
                _fnirs_n = len(_hbo_vals)

        _lines.append(
            f"**fNIRS** ({_fnirs_freq}{', ' + str(_fnirs_n) + ' post-stimulus samples' if _fnirs_n > 0 else ''})"
        )
        _lines.append("")
        _lines.append("| | Mean (uM) | Peak (uM) |")
        _lines.append("|---|---|---|")
        if _fnirs_n > 0:
            _lines.append(f"| HbO | {_hbo_mean:.4f} | {_hbo_peak:.4f} |")
            _lines.append(f"| HbR | {_hbr_mean:.4f} | {_hbr_peak:.4f} |")
        else:
            _lines.append("| HbO | \u2014 | \u2014 |")
            _lines.append("| HbR | \u2014 | \u2014 |")

        _lines.append("")

        # EMG stats
        _emg_n = 0
        if _r_emg_post.height > 0 and len(pr_emg_cols) > 0:
            _emg_n = _r_emg_post.height

        _lines.append(
            f"**EMG** ({_emg_freq}{', ' + f'{_emg_n:,}' + ' post-stimulus samples' if _emg_n > 0 else ''})"
        )
        _lines.append("")
        _lines.append("| Sensor | Mean (mV) | Peak (mV) |")
        _lines.append("|--------|-----------|-----------|")
        if _emg_n > 0:
            for _col, _lbl in zip(pr_emg_cols, pr_emg_labels):
                _vals = _r_emg_post[_col].drop_nulls()
                if len(_vals) > 0:
                    _m = _vals.mean()
                    _p = _vals.max()
                    _lines.append(f"| {_lbl} | {_m:.4f} | {_p:.4f} |")
                else:
                    _lines.append(f"| {_lbl} | \u2014 | \u2014 |")
        else:
            for _lbl in pr_emg_labels:
                _lines.append(f"| {_lbl} | \u2014 | \u2014 |")

    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(
    emg_df,
    fnirs_df,
    np,
    pl,
    plt,
    pr_baseline_switch,
    pr_robot_select,
    pr_run_select,
    pr_task_select,
):
    # Per-Run Viewer Plot
    # Reuses: fnirs_df, emg_df, pl, plt, np from existing cells

    pr_TIME_MIN = -5.0
    pr_TIME_MAX = 15.0

    pr_hbo_cols = [c for c in fnirs_df.columns if c.endswith("_hbo")]
    pr_hbr_cols = [c for c in fnirs_df.columns if c.endswith("_hbr")]
    pr_emg_cols = [c for c in emg_df.columns if "EMG" in c and c.endswith("(mV)")]

    # EMG labels — full sensor names
    pr_emg_labels = []
    for _c in pr_emg_cols:
        _parts = _c.split(" | ")
        _sensor_name = _parts[0]
        _emg_ch = _parts[1].replace(" (mV)", "") if len(_parts) > 1 else _c
        pr_emg_labels.append(f"{_sensor_name} {_emg_ch}")

    if pr_run_select.value is None or pr_task_select.value is None:
        pr_fig, _ax = plt.subplots(figsize=(14, 3))
        _ax.text(
            0.5,
            0.5,
            "Select a run and task to view",
            ha="center",
            va="center",
            fontsize=14,
        )
        _ax.set_axis_off()
        plt.show()

    else:
        _cond_filter = (
            (pl.col("run_id") == pr_run_select.value)
            & (pl.col("task") == pr_task_select.value)
            & (pl.col("is_robot") == (pr_robot_select.value == "Robot"))
            & (pl.col("time_sec") >= pr_TIME_MIN)
            & (pl.col("time_sec") <= pr_TIME_MAX)
        )

        _run_fnirs = fnirs_df.filter(_cond_filter).sort("time_sec")
        _run_emg = emg_df.filter(_cond_filter).sort("time_sec")

        if _run_fnirs.height == 0 and _run_emg.height == 0:
            pr_fig, _ax = plt.subplots(figsize=(14, 3))
            _ax.text(
                0.5,
                0.5,
                "No data matches this selection",
                ha="center",
                va="center",
                fontsize=14,
            )
            _ax.set_axis_off()
            plt.show()

        else:
            # ============================================================
            # fNIRS
            # ============================================================
            if (
                len(pr_hbo_cols) == 0
                or len(pr_hbr_cols) == 0
                or _run_fnirs.height == 0
            ):
                _has_fnirs = False
                _fnirs_time = np.array([])
                _hbo = np.array([])
                _hbr = np.array([])
            else:
                _has_fnirs = True

                _fnirs_df_plot = (
                    _run_fnirs.with_columns(
                        [
                            pl.mean_horizontal(pr_hbo_cols).alias("HbO"),
                            pl.mean_horizontal(pr_hbr_cols).alias("HbR"),
                        ]
                    )
                    .group_by("time_sec")
                    .agg(
                        [
                            pl.col("HbO").mean().alias("HbO"),
                            pl.col("HbR").mean().alias("HbR"),
                        ]
                    )
                    .sort("time_sec")
                )

                _fnirs_time = _fnirs_df_plot["time_sec"].to_numpy()
                _fnirs_bl = _fnirs_time < 0

                _hbo = _fnirs_df_plot["HbO"].to_numpy() * 1e6
                _hbr = _fnirs_df_plot["HbR"].to_numpy() * 1e6

                if pr_baseline_switch.value:
                    if np.any(_fnirs_bl):
                        _hbo_bl = np.nanmean(_hbo[_fnirs_bl])
                        _hbr_bl = np.nanmean(_hbr[_fnirs_bl])
                        if not np.isnan(_hbo_bl):
                            _hbo = _hbo - _hbo_bl
                        if not np.isnan(_hbr_bl):
                            _hbr = _hbr - _hbr_bl

            # ============================================================
            # EMG
            # ============================================================
            _emg_data = {}
            for _col in pr_emg_cols:
                _emg_s = (
                    _run_emg.select(["time_sec", _col])
                    .group_by("time_sec")
                    .agg(pl.col(_col).mean().alias(_col))
                    .sort("time_sec")
                )
                _time = _emg_s["time_sec"].to_numpy()
                _raw = _emg_s[_col].to_numpy().copy()
                _emg_bl = _time < 0
                if pr_baseline_switch.value:
                    _valid = ~np.isnan(_raw)
                    _bl_vals = _raw[_valid & _emg_bl]
                    if len(_bl_vals) > 0:
                        _raw[_valid] = _raw[_valid] - np.nanmean(_bl_vals)
                _emg_data[_col] = {"time": _time, "data": _raw}

            # ============================================================
            # Figure
            # ============================================================
            _n_emg = len(pr_emg_cols)
            _fig, _axes = plt.subplots(
                _n_emg + 1,
                1,
                figsize=(14, 2.5 * (_n_emg + 1)),
                sharex=True,
                gridspec_kw={"hspace": 0.15},
            )
            if _n_emg == 0:
                _axes = [_axes]

            # fNIRS plot
            if _has_fnirs and len(_fnirs_time) > 0:
                _axes[0].plot(
                    _fnirs_time, _hbo, label="HbO", color="red", linewidth=1.2
                )
                _axes[0].plot(
                    _fnirs_time, _hbr, label="HbR", color="blue", linewidth=1.2
                )
                _axes[0].set_ylabel("Concentration (μM)")
                _axes[0].legend(loc="upper right", fontsize=8)
            else:
                _axes[0].text(
                    0.5,
                    0.5,
                    "No fNIRS data",
                    ha="center",
                    va="center",
                    transform=_axes[0].transAxes,
                )
                _axes[0].set_ylabel("fNIRS")

            _axes[0].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
            _axes[0].set_title(
                f"fNIRS — {pr_run_select.value} — {pr_task_select.value} — {pr_robot_select.value}"
            )

            # EMG plots
            for i, (_col, _label) in enumerate(zip(pr_emg_cols, pr_emg_labels)):
                _axes[i + 1].plot(
                    _emg_data[_col]["time"],
                    _emg_data[_col]["data"],
                    color=f"C{i}",
                    linewidth=0.5,
                )
                _axes[i + 1].set_ylabel(f"{_label}\n(mV)", fontsize=8)
                _axes[i + 1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)

            _axes[-1].set_xlabel("Time (s)")
            for _ax in _axes:
                _ax.set_xlim(pr_TIME_MIN, pr_TIME_MAX)

            plt.tight_layout()
            plt.show()

    _fig
    return pr_emg_cols, pr_emg_labels, pr_hbo_cols, pr_hbr_cols


if __name__ == "__main__":
    app.run()
