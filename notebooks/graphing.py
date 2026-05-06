import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import numpy as np
    import sys
    sys.path.insert(0, "/Users/haider/code/Franka-fNIRS-EMG-data_analysis")
    from src.loaders.emg_filtering import filter_rms

    return filter_rms, mo, np, pl, plt


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
    - EMG: `{emg_df.shape[0]:,}` rows × {emg_df.shape[1]} cols (~1259 Hz)
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
def _(filter_rms, np, pl):
    # Reusable helpers for all plots
    # Reuses: pl, np from imports cell
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D


    def apply_baseline(df, signal_cols, time_col="time_sec", baseline_end=0):
        """Subtract per-run baseline (mean of time < baseline_end) from signal columns.
        Returns a new DataFrame with baseline-corrected columns."""
        _bl = (
            df.filter(pl.col(time_col) < baseline_end)
            .group_by(["run_id", "task_instance"])
            .agg(
                [
                    pl.col(c).drop_nulls().mean().alias(f"{c}_bl")
                    for c in signal_cols
                ]
            )
        )
        _result = df.join(_bl, on=["run_id", "task_instance"], how="left")
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


    def filter_rms_per_epoch(emg_df):
        """Apply filter_rms to each (run_id, task_instance) independently.
        Avoids cross-epoch artifacts from sosfiltfilt."""
        return pl.concat(
            [
                filter_rms(g.sort("time_sec"))
                for _, g in emg_df.group_by("run_id", "task_instance")
            ]
        )

    def filter_rms_per_epoch(emg_df):
        """
        Apply filter_rms independently to each EMG epoch.

        This prevents sosfiltfilt from filtering across task/run boundaries where
        time_sec resets or where the signal is discontinuous.
        """
        group_cols = ["run_id", "task_instance"]

        filtered_epochs = []

        for _, g in emg_df.group_by(group_cols, maintain_order=True):
            g = g.sort("time_sec")

            # sosfiltfilt needs enough samples; skip tiny/bad epochs safely
            if g.height < 20:
                filtered_epochs.append(g)
                continue

            filtered_epochs.append(filter_rms(g))

        return (
            pl.concat(filtered_epochs, how="vertical")
            .sort(group_cols + ["time_sec"])
        )

    def align_emg_epochs_to_common_time(
        emg_df,
        signal_cols,
        time_min=-5.0,
        time_max=15.0,
        group_cols=("run_id", "task_instance"),
    ):
        """
        Align EMG epochs to a common time grid without downsampling.

        Uses the median native sampling interval from the data, then interpolates
        each epoch onto the same time_sec vector.
        """
        # Estimate native dt across epochs
        dt_df = (
            emg_df
            .sort(list(group_cols) + ["time_sec"])
            .with_columns(
                pl.col("time_sec")
                .diff()
                .over(list(group_cols))
                .alias("_dt")
            )
        )

        dt = dt_df["_dt"].drop_nulls().median()
        fs = 1.0 / dt

        # Common native-rate grid
        t_grid = np.arange(time_min, time_max, dt)

        aligned = []

        meta_cols = [
            c for c in emg_df.columns
            if c not in signal_cols and c != "time_sec"
        ]

        for key, g in emg_df.group_by(list(group_cols), maintain_order=True):
            g = g.sort("time_sec")

            t = g["time_sec"].to_numpy()

            row_dict = {
                "time_sec": t_grid,
            }

            # Preserve grouping columns
            if len(group_cols) == 1:
                key = (key,)

            for col_name, value in zip(group_cols, key):
                row_dict[col_name] = value

            # Preserve useful metadata as first value in epoch
            for c in meta_cols:
                if c not in group_cols:
                    row_dict[c] = g[c][0]

            # Interpolate each EMG channel
            for c in signal_cols:
                y = g[c].to_numpy()
                valid = np.isfinite(t) & np.isfinite(y)

                if valid.sum() < 2:
                    row_dict[c] = np.full_like(t_grid, np.nan, dtype=float)
                else:
                    row_dict[c] = np.interp(
                        t_grid,
                        t[valid],
                        y[valid],
                        left=np.nan,
                        right=np.nan,
                    )

            aligned.append(pl.DataFrame(row_dict))

        return pl.concat(aligned, how="vertical")

    def prepare_emg_for_analysis(
        emg_df,
        apply_filter=False,
        time_min=-5.0,
        time_max=15.0,
    ):
        """
        Prepare EMG for all plotting/statistics.

        - Optionally applies bandpass + RMS per epoch.
        - Aligns every epoch to the same native-rate time grid.
        - Returns a dataframe safe to group/average by exact time_sec.
        """

        emg_cols = [
            c for c in emg_df.columns
            if "EMG" in c and c.endswith("(mV)")
        ]

        if apply_filter:
            out = filter_rms_per_epoch(emg_df)
        else:
            out = emg_df

        out = align_emg_epochs_to_common_time(
            out,
            signal_cols=emg_cols,
            time_min=time_min,
            time_max=time_max,
            group_cols=("run_id", "task_instance"),
        )

        out = out.with_columns(
        [
            pl.when(pl.col(c).is_nan())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in emg_cols
        ]
    )

        return out

    return (
        Line2D,
        apply_baseline,
        build_legend,
        legend_layout,
        prepare_emg_for_analysis,
    )


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
                .group_by(["run_id", "task_instance"])
                .agg(pl.col(_col).mean().alias("_base"))
            )
            fnirs_run_avg = (
                fnirs_run_avg.join(_bl, on=["run_id", "task_instance"], how="left")
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
def _(fnirs_baseline_switch, fnirs_df, mo, pl):
    # fNIRS Overall Summary Stats
    # True condition-level result:
    # collect all epochs for each condition and average them together
    # Reuses: fnirs_df, pl, mo from existing cells

    _fnirs_ov_TIME_MIN = -5.0
    _fnirs_ov_TIME_MAX = 15.0

    _fnirs_ov_filtered = fnirs_df.filter(
        (pl.col("time_sec") >= _fnirs_ov_TIME_MIN)
        & (pl.col("time_sec") <= _fnirs_ov_TIME_MAX)
    )

    _fnirs_ov_hbo = [c for c in fnirs_df.columns if c.endswith("_hbo")]
    _fnirs_ov_hbr = [c for c in fnirs_df.columns if c.endswith("_hbr")]

    # Average HbO and HbR channels within each epoch sample
    _fnirs_ov_epoch = _fnirs_ov_filtered.with_columns(
        [
            pl.mean_horizontal(_fnirs_ov_hbo).alias("hbo_mean"),
            pl.mean_horizontal(_fnirs_ov_hbr).alias("hbr_mean"),
        ]
    )

    if fnirs_baseline_switch.value:
        for _col in ["hbo_mean", "hbr_mean"]:
            _bl = (
                _fnirs_ov_epoch.filter(pl.col("time_sec") < 0)
                .group_by(["run_id", "task_instance"])
                .agg(pl.col(_col).mean().alias("_base"))
            )
            _fnirs_ov_epoch = (
                _fnirs_ov_epoch.join(_bl, on=["run_id", "task_instance"], how="left")
                .with_columns((pl.col(_col) - pl.col("_base")).alias(_col))
                .drop("_base")
            )

    # Condition-level average curves:
    # all epochs in each condition averaged together at each time_sec
    _fnirs_ov_condition_curve = (
        _fnirs_ov_epoch
        .group_by(["is_robot", "time_sec"])
        .agg(
            [
                pl.col("hbo_mean").mean().alias("hbo"),
                pl.col("hbr_mean").mean().alias("hbr"),
            ]
        )
        .sort(["is_robot", "time_sec"])
    )

    _fnirs_ov_lines = []

    for _rob, _cond in [(True, "Robot"), (False, "No-Robot")]:
        _epoch_df = _fnirs_ov_epoch.filter(pl.col("is_robot") == _rob)

        # If run_id + task_instance defines an epoch, use this.
        # If fNIRS only has one row per run_id/task, this is still safer than run_id alone.
        _epochs = _epoch_df.select(["run_id", "task_instance"]).unique().height \
            if "task_instance" in _epoch_df.columns \
            else _epoch_df["run_id"].n_unique()

        _participants = _epoch_df["participant"].n_unique()
        _tasks = _epoch_df["task"].n_unique()

        _curve = _fnirs_ov_condition_curve.filter(pl.col("is_robot") == _rob)

        # Post-stimulus condition-level curve: 0–15s
        _post = _curve.filter(
            (pl.col("time_sec") >= 0)
            & (pl.col("time_sec") <= 15.0)
        )

        # Baseline condition-level curve: -5–0s
        _baseline = _curve.filter(
            (pl.col("time_sec") >= -5.0)
            & (pl.col("time_sec") < 0)
        )

        # Convert to μM after computing values
        _hbo_m = _post["hbo"].mean() * 1e6
        _hbo_x = _post["hbo"].max() * 1e6
        _hbo_bl = _baseline["hbo"].mean() * 1e6 if _baseline.height > 0 else 0.0

        _hbr_m = _post["hbr"].mean() * 1e6
        _hbr_x = _post["hbr"].max() * 1e6
        _hbr_bl = _baseline["hbr"].mean() * 1e6 if _baseline.height > 0 else 0.0

        _fnirs_ov_lines.append(
            f"| {_cond} | {_epochs} | {_participants} | {_tasks} | "
            f"{_hbo_m:.3f} | {_hbo_x:.3f} | {_hbo_bl:.3f} | "
            f"{_hbr_m:.3f} | {_hbr_x:.3f} | {_hbr_bl:.3f} |"
        )

    _fnirs_ov_tbl = (
        "| Condition | Epochs | Participants | Tasks | "
        "HbO Mean Curve 0–15s (μM) | HbO Max Curve 0–15s (μM) | HbO Baseline −5–0s (μM) | "
        "HbR Mean Curve 0–15s (μM) | HbR Max Curve 0–15s (μM) | HbR Baseline −5–0s (μM) |\n"
    )
    _fnirs_ov_tbl += (
        "|-----------|--------|--------------|-------|"
        "----------------------------|---------------------------|--------------------------|"
        "----------------------------|---------------------------|--------------------------|\n"
    )
    _fnirs_ov_tbl += "\n".join(_fnirs_ov_lines)

    mo.md(f"""
    ### fNIRS Overall Summary

    This summary matches a condition-level averaged fNIRS graph.

    For each condition, all epochs are pooled and averaged together at each `time_sec`. HbO and HbR channels are averaged within each epoch sample first, then all epochs in the condition are averaged at each time point.

    Mean and max are computed from the condition-level averaged curve over 0–15s post-stimulus. Baseline is computed from the same condition-level averaged curve over −5 to 0s.

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
def _(fnirs_baseline_switch, fnirs_channel_options, fnirs_channel_selector, fnirs_df, mo, pl):
    # fNIRS Channel Plot Summary Stats
    # Matches: fNIRS Per-Channel Interactive Plot logic

    _fnirs_ch_statsFiltered = fnirs_df.filter(
        (pl.col("time_sec") >= -5.0) & (pl.col("time_sec") <= 15.0)
    )

    _fnirs_ch_selected = [
        fnirs_channel_options[_ch] for _ch in fnirs_channel_selector.value
    ]

    # Baseline correction (matching graph logic)
    if fnirs_baseline_switch.value and _fnirs_ch_selected:
        for _col in _fnirs_ch_selected:
            _bl = (
                _fnirs_ch_statsFiltered.filter(pl.col("time_sec") < 0)
                .group_by(["run_id", "task_instance"])
                .agg(pl.col(_col).mean().alias("_base"))
            )
            _fnirs_ch_statsFiltered = (
                _fnirs_ch_statsFiltered.join(_bl, on=["run_id", "task_instance"], how="left")
                .with_columns((pl.col(_col) - pl.col("_base")).alias(_col))
                .drop("_base")
            )

    _fnirs_ch_lines = []
    for _lbl in fnirs_channel_selector.value:
        _col = fnirs_channel_options[_lbl]
        _pair = _lbl.replace(" HbO", "").replace(" HbR", "")
        _chrom = "HbO" if "HbO" in _lbl else "HbR"
        for _rob, _cond in [(True, "Robot"), (False, "No-Robot")]:
            # Compute the averaged curve (matching graph: group_by time_sec → mean)
            _curve = (
                _fnirs_ch_statsFiltered.filter(pl.col("is_robot") == _rob)
                .group_by("time_sec")
                .agg(pl.col(_col).mean().alias("val"))
                .sort("time_sec")
                .with_columns((pl.col("val") * 1e6).alias("val"))
            )
            _curve_post = _curve.filter(pl.col("time_sec") >= 0)
            _curve_bl = _curve.filter(pl.col("time_sec") < 0)

            if _curve_post.height == 0:
                continue

            _vals_mean = _curve_post["val"].mean()
            _vals_max = _curve_post["val"].max()
            _bl_mean = _curve_bl["val"].mean() if _curve_bl.height > 0 else 0.0
            _fnirs_ch_lines.append(
                f"| {_pair} | {_chrom} | {_cond} | {_vals_mean:.3f} | {_vals_max:.3f} | {_bl_mean:.3f} |"
            )

    _fnirs_ch_tbl = (
        "| Channel | Chrom | Condition | Mean (μM) | Max (μM) | Baseline (μM) |\n"
    )
    _fnirs_ch_tbl += (
        "|---------|-------|-----------|-----------|----------|---------------|\n"
    )
    _fnirs_ch_tbl += "\n".join(_fnirs_ch_lines)

    _fn = " with baseline correction" if fnirs_baseline_switch.value else ""

    mo.md(f"""
    ### fNIRS Per-Channel Summary{_fn}

    Mean and max concentration (μM) per source-detector channel, from the condition-level averaged curve (matching the graph).

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
                .group_by(["run_id", "task_instance"])
                .agg(pl.col(_col).mean().alias("_base"))
            )
            fnirs_ch_filtered = (
                fnirs_ch_filtered.join(_bl, on=["run_id", "task_instance"], how="left")
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
def _(fnirs_baseline_switch, fnirs_df, fnirs_task_selector, mo, pl):
    # fNIRS Task Plot Summary Stats
    # Matches: fNIRS Per-Task Interactive Plot logic

    _fnirs_task_statsFiltered = fnirs_df.filter(
        (pl.col("time_sec") >= -5.0) & (pl.col("time_sec") <= 15.0)
    )
    _fnirs_task_hbo_cols_s = [c for c in fnirs_df.columns if c.endswith("_hbo")]
    _fnirs_task_hbr_cols_s = [c for c in fnirs_df.columns if c.endswith("_hbr")]

    _fnirs_task_lines = []
    for _task in fnirs_task_selector.value:
        _tdf = _fnirs_task_statsFiltered.filter(pl.col("task") == _task)

        for _rob, _lbl in [(True, "Robot"), (False, "No-Robot")]:
            _cdf = _tdf.filter(pl.col("is_robot") == _rob)
            _runs = _cdf.select("run_id", "task_instance").unique().height
            if _runs == 0:
                continue

            # Average channels → hbo_mean, hbr_mean (matching graph step 1)
            _cdf_avg = _cdf.with_columns(
                [
                    pl.mean_horizontal(_fnirs_task_hbo_cols_s).alias("hbo_mean"),
                    pl.mean_horizontal(_fnirs_task_hbr_cols_s).alias("hbr_mean"),
                ]
            )

            # Baseline correction (matching graph step 2)
            if fnirs_baseline_switch.value:
                for _col in ["hbo_mean", "hbr_mean"]:
                    _bl = (
                        _cdf_avg.filter(pl.col("time_sec") < 0)
                        .group_by(["run_id", "task_instance"])
                        .agg(pl.col(_col).mean().alias("_base"))
                    )
                    _cdf_avg = (
                        _cdf_avg.join(_bl, on=["run_id", "task_instance"], how="left")
                        .with_columns((pl.col(_col) - pl.col("_base")).alias(_col))
                        .drop("_base")
                    )

            # Average across runs per time_sec (matching graph step 3)
            _task_curve = (
                _cdf_avg.group_by("time_sec")
                .agg(
                    [
                        pl.col("hbo_mean").mean().alias("hbo"),
                        pl.col("hbr_mean").mean().alias("hbr"),
                    ]
                )
                .sort("time_sec")
            )

            # Post-stimulus: 0–15s
            _post = _task_curve.filter(
                (pl.col("time_sec") >= 0) & (pl.col("time_sec") <= 15.0)
            )
            # Baseline: -5–0s
            _baseline = _task_curve.filter(
                (pl.col("time_sec") >= -5.0) & (pl.col("time_sec") < 0)
            )

            # Convert to μM
            _hbo_m = _post["hbo"].mean() * 1e6 if _post.height > 0 else 0.0
            _hbo_x = _post["hbo"].max() * 1e6 if _post.height > 0 else 0.0
            _hbr_m = _post["hbr"].mean() * 1e6 if _post.height > 0 else 0.0
            _hbr_x = _post["hbr"].max() * 1e6 if _post.height > 0 else 0.0
            _hbo_bl = _baseline["hbo"].mean() * 1e6 if _baseline.height > 0 else 0.0
            _hbr_bl = _baseline["hbr"].mean() * 1e6 if _baseline.height > 0 else 0.0

            _fnirs_task_lines.append(
                f"| {_task} | {_lbl} | {_runs} | {_hbo_m:.3f} | {_hbo_x:.3f} | {_hbo_bl:.3f} | {_hbr_m:.3f} | {_hbr_x:.3f} | {_hbr_bl:.3f} |"
            )

    _fnirs_task_tbl = "| Task | Condition | Runs | HbO Mean (μM) | HbO Max (μM) | HbO Baseline (μM) | HbR Mean (μM) | HbR Max (μM) | HbR Baseline (μM) |\n"
    _fnirs_task_tbl += "|------|-----------|------|---------------|--------------|-------------------|---------------|--------------|-------------------|\n"
    _fnirs_task_tbl += "\n".join(_fnirs_task_lines)

    _fn = " with baseline correction" if fnirs_baseline_switch.value else ""

    mo.md(f"""
    ### fNIRS Per-Task Summary{_fn}

    Mean and max concentration (μM) of the global channel average, from the condition-level averaged curve (matching the graph).

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



    fnirs_task_fig
    return


@app.cell(hide_code=True)
def emg_baseline_switch(mo):
    # ─── EMG Plots ─────────────────────────────────────────────
    emg_baseline_switch = mo.ui.switch(label="Baseline correction (-5 to 0s)")
    emg_ov_filter_switch = mo.ui.switch(
        label="Bandpass + RMS filter (20-450 Hz, 100 ms)"
    )
    [emg_baseline_switch, emg_ov_filter_switch]
    return emg_baseline_switch, emg_ov_filter_switch


@app.cell(hide_code=True)
def _(
    emg_baseline_switch,
    emg_df,
    emg_ov_filter_switch,
    pl,
    plt,
    prepare_emg_for_analysis,
):
    # EMG Aggregated: Robot vs No-Robot (Global Sensor Average)
    # Reuses: emg_df, pl, plt, np, filter_rms, emg_ov_filter_switch

    _emg_src = prepare_emg_for_analysis(
        emg_df,
        apply_filter=emg_ov_filter_switch.value,
        time_min=-5.0,
        time_max=15.0,
    )
    _ylabel = "EMG RMS (mV)" if emg_ov_filter_switch.value else "EMG (mV)"

    _emg_ov_TIME_MIN = -5.0
    _emg_ov_TIME_MAX = 15.0

    _emg_ov_cols = [
        c for c in _emg_src.columns if "EMG" in c and c.endswith("(mV)")
    ]

    _emg_ov_filtered = _emg_src.filter(
        (pl.col("time_sec") >= _emg_ov_TIME_MIN)
        & (pl.col("time_sec") <= _emg_ov_TIME_MAX)
    )

    _emg_ov_run_avg = _emg_ov_filtered.with_columns(
        pl.mean_horizontal(_emg_ov_cols).alias("emg_mean")
    )

    # Baseline correction
    if emg_baseline_switch.value:
        for _col in _emg_ov_cols:
            _bl = (
                _emg_ov_run_avg.filter(pl.col("time_sec") < 0)
                .group_by(["run_id", "task_instance"])
                .agg(pl.col(_col).mean().alias("_base"))
            )
            _emg_ov_run_avg = (
                _emg_ov_run_avg.join(_bl, on=["run_id", "task_instance"], how="left")
                .with_columns((pl.col(_col) - pl.col("_base")).alias(_col))
                .drop("_base")
            )
        _emg_ov_run_avg = _emg_ov_run_avg.with_columns(
            pl.mean_horizontal(_emg_ov_cols).alias("emg_mean")
        )

    _emg_ov_avg = (
        _emg_ov_run_avg.group_by("time_sec", "is_robot")
        .agg(pl.col("emg_mean").mean().alias("emg"))
        .sort("time_sec")
    )

    _emg_ov_robot = _emg_ov_avg.filter(pl.col("is_robot") == True)
    _emg_ov_no_robot = _emg_ov_avg.filter(pl.col("is_robot") == False)

    _emg_ov_fig, (_emg_ov_ax1, _emg_ov_ax2) = plt.subplots(1, 2, figsize=(14, 5))

    _emg_ov_ax1.plot(
        _emg_ov_robot["time_sec"],
        _emg_ov_robot["emg"],
        color="tab:blue",
        linewidth=0.8,
    )
    _emg_ov_ax1.set_title("Robot Trials")
    _emg_ov_ax1.set_xlabel("Time (s)")
    _emg_ov_ax1.set_ylabel(_ylabel)
    _emg_ov_ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    _emg_ov_ax2.plot(
        _emg_ov_no_robot["time_sec"],
        _emg_ov_no_robot["emg"],
        color="tab:orange",
        linewidth=0.8,
    )
    _emg_ov_ax2.set_title("No-Robot Trials")
    _emg_ov_ax2.set_xlabel("Time (s)")
    _emg_ov_ax2.set_ylabel(_ylabel)
    _emg_ov_ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()


    _emg_ov_fig
    return


@app.cell(hide_code=True)
def _(
    emg_baseline_switch,
    emg_df,
    emg_ov_filter_switch,
    mo,
    pl,
    prepare_emg_for_analysis,
):
    # EMG Overall Summary Stats
    # Matches: EMG Aggregated Robot vs No-Robot graph logic
    # Reuses: emg_df, pl, mo, filter_rms, emg_ov_filter_switch, emg_baseline_switch

    _emg_src = prepare_emg_for_analysis(
        emg_df,
        apply_filter=emg_ov_filter_switch.value,
        time_min=-5.0,
        time_max=15.0,
    )

    _emg_ov_TIME_MIN = -5.0
    _emg_ov_TIME_MAX = 15.0

    _emg_ov_cols = [
        c for c in _emg_src.columns
        if "EMG" in c and c.endswith("(mV)")
    ]

    _emg_ov_filtered = _emg_src.filter(
        (pl.col("time_sec") >= _emg_ov_TIME_MIN)
        & (pl.col("time_sec") <= _emg_ov_TIME_MAX)
    )

    # Per-row/global sensor average
    _emg_ov_run_avg = _emg_ov_filtered.with_columns(
        pl.mean_horizontal(_emg_ov_cols).alias("emg_mean")
    )

    # Match graph baseline correction exactly
    if emg_baseline_switch.value:
        for _col in _emg_ov_cols:
            _bl = (
                _emg_ov_run_avg
                .filter(pl.col("time_sec") < 0)
                .group_by(["run_id", "task_instance"])
                .agg(pl.col(_col).mean().alias("_base"))
            )

            _emg_ov_run_avg = (
                _emg_ov_run_avg
                .join(_bl, on=["run_id", "task_instance"], how="left")
                .with_columns(
                    (pl.col(_col) - pl.col("_base")).alias(_col)
                )
                .drop("_base")
            )

        _emg_ov_run_avg = _emg_ov_run_avg.with_columns(
            pl.mean_horizontal(_emg_ov_cols).alias("emg_mean")
        )

    # This is the exact condition-level curve used in the graph
    _emg_ov_avg = (
        _emg_ov_run_avg
        .group_by("time_sec", "is_robot")
        .agg(pl.col("emg_mean").mean().alias("emg"))
        .sort("time_sec")
    )

    _emg_ov_lines = []

    for _rob, _cond in [(True, "Robot"), (False, "No-Robot")]:
        _cdf = _emg_ov_run_avg.filter(pl.col("is_robot") == _rob)

        _runs = _cdf.select("run_id", "task_instance").unique().height
        _parts = _cdf["participant"].n_unique()
        _tasks = _cdf["task"].n_unique()

        # Same plotted curve for this condition
        _curve = _emg_ov_avg.filter(pl.col("is_robot") == _rob)

        # Post-stimulus graph values: 0–15s
        _curve_post = _curve.filter(
            (pl.col("time_sec") >= 0)
            & (pl.col("time_sec") <= 15.0)
        )

        # Baseline graph values: -5–0s
        _curve_bl = _curve.filter(
            (pl.col("time_sec") >= -5.0)
            & (pl.col("time_sec") < 0)
        )

        # These now match what the graph visually shows
        _emg_m = _curve_post["emg"].mean()
        _emg_x = _curve_post["emg"].max()
        _emg_bl = _curve_bl["emg"].mean() if _curve_bl.height > 0 else 0.0

        _emg_ov_lines.append(
            f"| {_cond} | {_runs} | {_parts} | {_tasks} | "
            f"{_emg_m:.6f} | {_emg_x:.6f} | {_emg_bl:.6f} |"
        )

    _emg_ov_tbl = (
        "| Condition | Epochs | Participants | Tasks | "
        "Mean of Epoch-Averaged Curve (mV) | Max of Epoch-Averaged Curve (mV) | Baseline of Epoch-Averaged Curve (mV) |\n"
    )
    _emg_ov_tbl += (
        "|-----------|--------|--------------|-------|"
        "-----------------------------------|----------------------------------|--------------------------------------|\n"
    )
    _emg_ov_tbl += "\n".join(_emg_ov_lines)

    _fn = " (bandpass + RMS)" if emg_ov_filter_switch.value else ""
    _bl_note = " with per-epoch baseline correction" if emg_baseline_switch.value else ""

    mo.md(f"""
    ### EMG Overall Summary{_fn}

    Values match the aggregated Robot vs No-Robot graph{_bl_note}. Each curve is computed by first averaging EMG sensors within each epoch sample, then averaging those epoch values at each time point by condition.

    Mean and max are computed from the plotted condition-level curve over 0–15s post-stimulus. Baseline is computed from the plotted condition-level curve over −5 to 0s.

    {_emg_ov_tbl}
    """)
    return


@app.cell(hide_code=True)
def _(emg_df, mo):
    # EMG Sensor Selector Widget + Filter Toggle
    # Reuses: mo, emg_df from existing cells

    _emg_sensor_options = {}
    for _c in emg_df.columns:
        if "EMG" in _c and _c.endswith("(mV)"):
            _parts = _c.split(" | ")
            _sensor_full = _parts[0]
            _emg_ch = _parts[1].replace(" (mV)", "")
            _words = _sensor_full.split()
            _short = f"{_words[0]} {_words[2]} ({_words[3].strip('()')})"
            if "Duo" in _sensor_full:
                _label = f"{_short} {_emg_ch}"
            else:
                _label = _short
            _emg_sensor_options[_label] = _c

    emg_sensor_options = _emg_sensor_options

    emg_sensor_selector = mo.ui.multiselect(
        options=list(emg_sensor_options.keys()),
        value=list(emg_sensor_options.keys()),
        label="Select EMG sensors",
    )
    emg_sens_filter_switch = mo.ui.switch(
        label="Bandpass + RMS filter (20-450 Hz, 100 ms)"
    )
    [emg_sensor_selector, emg_sens_filter_switch]
    return emg_sens_filter_switch, emg_sensor_options, emg_sensor_selector


@app.cell(hide_code=True)
def _(
    emg_baseline_switch,
    emg_df,
    emg_sens_filter_switch,
    emg_sensor_options,
    emg_sensor_selector,
    mo,
    pl,
    prepare_emg_for_analysis,
):
    # EMG Sensor Plot Summary Stats
    # Matches: EMG Per-Sensor Interactive Plot logic

    _emg_src = prepare_emg_for_analysis(
        emg_df,
        apply_filter=emg_sens_filter_switch.value,
        time_min=-5.0,
        time_max=15.0,
    )

    _emg_ss_sf = _emg_src.filter(
        (pl.col("time_sec") >= -5.0) & (pl.col("time_sec") <= 15.0)
    )

    # Baseline correction (matching graph logic)
    _emg_ss_selected = [
        emg_sensor_options[_ch] for _ch in emg_sensor_selector.value
    ]
    if emg_baseline_switch.value and _emg_ss_selected:
        for _col in _emg_ss_selected:
            _bl = (
                _emg_ss_sf.filter(pl.col("time_sec") < 0)
                .group_by(["run_id", "task_instance"])
                .agg(pl.col(_col).mean().alias("_base"))
            )
            _emg_ss_sf = (
                _emg_ss_sf.join(_bl, on=["run_id", "task_instance"], how="left")
                .with_columns((pl.col(_col) - pl.col("_base")).alias(_col))
                .drop("_base")
            )

    _emg_ss_lines = []
    for _lbl in emg_sensor_selector.value:
        _col = emg_sensor_options[_lbl]
        for _rob, _cond in [(True, "Robot"), (False, "No-Robot")]:
            # Compute the averaged curve (matching graph: group_by time_sec → mean)
            _curve = (
                _emg_ss_sf.filter(pl.col("is_robot") == _rob)
                .group_by("time_sec")
                .agg(pl.col(_col).mean().alias("val"))
                .sort("time_sec")
            )
            _curve_post = _curve.filter(pl.col("time_sec") >= 0)
            _curve_bl = _curve.filter(pl.col("time_sec") < 0)

            if _curve_post.height == 0:
                continue

            _vals_mean = _curve_post["val"].mean()
            _vals_max = _curve_post["val"].max()
            _bl_mean = _curve_bl["val"].mean() if _curve_bl.height > 0 else 0.0
            _emg_ss_lines.append(
                f"| {_lbl} | {_cond} | {_vals_mean:.4f} | {_vals_max:.4f} | {_bl_mean:.4f} |"
            )

    _emg_ss_tbl = "| Sensor | Condition | Mean (mV) | Max (mV) | Baseline (mV) |\n"
    _emg_ss_tbl += (
        "|--------|-----------|-----------|----------|---------------|\n"
    )
    _emg_ss_tbl += "\n".join(_emg_ss_lines)

    _fn = " (bandpass + RMS)" if emg_sens_filter_switch.value else ""
    _bl = " with baseline correction" if emg_baseline_switch.value else ""

    mo.md(f"""
    ### EMG Per-Sensor Summary{_fn}{_bl}

    Mean and max EMG (mV) per sensor, from the condition-level averaged curve (matching the graph).

    {_emg_ss_tbl}
    """)
    return


@app.cell(hide_code=True)
def _(
    Line2D,
    build_legend,
    emg_baseline_switch,
    emg_df,
    emg_sens_filter_switch,
    emg_sensor_options,
    emg_sensor_selector,
    legend_layout,
    pl,
    plt,
    prepare_emg_for_analysis,
):
    # EMG Per-Sensor Interactive Plot
    # Reuses: emg_df, pl, plt, np, Line2D, emg_sensor_selector, emg_sensor_options, filter_rms, emg_sens_filter_switch

    _emg_src = prepare_emg_for_analysis(
        emg_df,
        apply_filter=emg_sens_filter_switch.value,
        time_min=-5.0,
        time_max=15.0,
    )
    _ylabel = "EMG RMS (mV)" if emg_sens_filter_switch.value else "EMG (mV)"

    _emg_sp_TIME_MIN = -5.0
    _emg_sp_TIME_MAX = 15.0

    _emg_sp_filtered = _emg_src.filter(
        (pl.col("time_sec") >= _emg_sp_TIME_MIN)
        & (pl.col("time_sec") <= _emg_sp_TIME_MAX)
    )

    _emg_sp_selected = [
        emg_sensor_options[_ch] for _ch in emg_sensor_selector.value
    ]

    # Baseline correction
    if emg_baseline_switch.value:
        for _col in _emg_sp_selected:
            _bl = (
                _emg_sp_filtered.filter(pl.col("time_sec") < 0)
                .group_by(["run_id", "task_instance"])
                .agg(pl.col(_col).mean().alias("_base"))
            )
            _emg_sp_filtered = (
                _emg_sp_filtered.join(_bl, on=["run_id", "task_instance"], how="left")
                .with_columns((pl.col(_col) - pl.col("_base")).alias(_col))
                .drop("_base")
            )

    if not _emg_sp_selected:
        _emg_sp_fig, _ax = plt.subplots(figsize=(14, 5))
        _ax.text(
            0.5, 0.5, "No sensors selected", ha="center", va="center", fontsize=14
        )
        _ax.set_axis_off()

    else:
        _emg_sp_robot = {}
        _emg_sp_no_robot = {}

        for _col in _emg_sp_selected:
            _avg = (
                _emg_sp_filtered.group_by("time_sec", "is_robot")
                .agg(pl.col(_col).mean().alias("val"))
                .sort("time_sec")
            )
            _emg_sp_robot[_col] = _avg.filter(pl.col("is_robot") == True)
            _emg_sp_no_robot[_col] = _avg.filter(pl.col("is_robot") == False)

        _n_sens = len(_emg_sp_selected)
        _cmap = plt.cm.tab10 if _n_sens <= 10 else plt.cm.tab20
        _emg_sp_colors = {
            _c: _cmap(_i / max(_n_sens - 1, 1))
            for _i, _c in enumerate(_emg_sp_selected)
        }

        _emg_sp_fig, (_emg_sp_ax1, _emg_sp_ax2) = plt.subplots(
            1, 2, figsize=(14, 5)
        )

        for _lbl in emg_sensor_selector.value:
            _col = emg_sensor_options[_lbl]
            _c = _emg_sp_colors[_col]
            _emg_sp_ax1.plot(
                _emg_sp_robot[_col]["time_sec"].to_numpy(),
                _emg_sp_robot[_col]["val"].to_numpy(),
                color=_c,
                label=_lbl,
                linewidth=0.8,
            )
            _emg_sp_ax2.plot(
                _emg_sp_no_robot[_col]["time_sec"].to_numpy(),
                _emg_sp_no_robot[_col]["val"].to_numpy(),
                color=_c,
                label=_lbl,
                linewidth=0.8,
            )

        _emg_sp_ax1.set_title("Robot Trials")
        _emg_sp_ax1.set_xlabel("Time (s)")
        _emg_sp_ax1.set_ylabel(_ylabel)
        _emg_sp_ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        _emg_sp_ax2.set_title("No-Robot Trials")
        _emg_sp_ax2.set_xlabel("Time (s)")
        _emg_sp_ax2.set_ylabel(_ylabel)
        _emg_sp_ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        _legend_handles = []
        for _lbl in emg_sensor_selector.value:
            _col = emg_sensor_options[_lbl]
            _c = _emg_sp_colors[_col]
            _legend_handles.append(Line2D([0], [0], color=_c, label=_lbl))

        legend_layout(_emg_sp_fig, len(_legend_handles))
        build_legend(_emg_sp_fig, _legend_handles)



    _emg_sp_fig
    return


@app.cell(hide_code=True)
def _(emg_df, mo):
    # EMG Task Selector Widget + Filter Toggle
    # Reuses: mo, emg_df from existing cells

    _emg_task_sorted = sorted(
        emg_df["task"].unique().to_list(), key=lambda _x: int(_x.split("_")[1])
    )

    emg_task_selector = mo.ui.multiselect(
        options=_emg_task_sorted, value=_emg_task_sorted, label="Select tasks"
    )
    emg_task_filter_switch = mo.ui.switch(
        label="Bandpass + RMS filter (20-450 Hz, 100 ms)"
    )
    [emg_task_selector, emg_task_filter_switch]
    return emg_task_filter_switch, emg_task_selector


@app.cell(hide_code=True)
def _(
    emg_baseline_switch,
    emg_df,
    emg_task_filter_switch,
    emg_task_selector,
    mo,
    pl,
    prepare_emg_for_analysis,
):
    # EMG Task Plot Summary Stats
    # Matches: EMG Per-Task Interactive Plot logic

    _emg_src = prepare_emg_for_analysis(
        emg_df,
        apply_filter=emg_task_filter_switch.value,
        time_min=-5.0,
        time_max=15.0,
    )

    _emg_ts_sf = _emg_src.filter(
        (pl.col("time_sec") >= -5.0) & (pl.col("time_sec") <= 15.0)
    )
    _emg_ts_ec = [c for c in _emg_src.columns if "EMG" in c and c.endswith("(mV)")]

    _emg_ts_lines = []
    for _task in emg_task_selector.value:
        _tdf = _emg_ts_sf.filter(pl.col("task") == _task)

        for _rob, _lbl in [(True, "Robot"), (False, "No-Robot")]:
            _cdf = _tdf.filter(pl.col("is_robot") == _rob)
            _epochs = _cdf.select("run_id", "task_instance").unique().height
            if _epochs == 0:
                continue

            # Average sensors → emg_mean (matching graph step 1)
            _cdf_avg = _cdf.with_columns(
                pl.mean_horizontal(_emg_ts_ec).alias("emg_mean")
            )

            # Baseline correction (matching graph step 2)
            if emg_baseline_switch.value:
                _bl = (
                    _cdf_avg.filter(pl.col("time_sec") < 0)
                    .group_by(["run_id", "task_instance"])
                    .agg(pl.col("emg_mean").mean().alias("_base"))
                )
                _cdf_avg = (
                    _cdf_avg.join(_bl, on=["run_id", "task_instance"], how="left")
                    .with_columns((pl.col("emg_mean") - pl.col("_base")).alias("emg_mean"))
                    .drop("_base")
                )

            # Average across runs per time_sec (matching graph step 3)
            _task_curve = (
                _cdf_avg.group_by("time_sec")
                .agg(pl.col("emg_mean").mean().alias("emg"))
                .sort("time_sec")
            )

            # Post-stimulus: 0–15s
            _post = _task_curve.filter(
                (pl.col("time_sec") >= 0) & (pl.col("time_sec") <= 15.0)
            )
            # Baseline: -5–0s
            _baseline = _task_curve.filter(
                (pl.col("time_sec") >= -5.0) & (pl.col("time_sec") < 0)
            )

            _emg_m = _post["emg"].mean() if _post.height > 0 else 0.0
            _emg_x = _post["emg"].max() if _post.height > 0 else 0.0
            _emg_bl = _baseline["emg"].mean() if _baseline.height > 0 else 0.0

            _emg_ts_lines.append(
                f"| {_task} | {_lbl} | {_epochs} | {_emg_m:.4f} | {_emg_x:.4f} | {_emg_bl:.4f} |"
            )

    _emg_ts_tbl = "| Task | Condition | Epochs | EMG Mean (mV) | EMG Max (mV) | EMG Baseline (mV) |\n"
    _emg_ts_tbl += "|------|-----------|--------|---------------|--------------|-------------------|\n"
    _emg_ts_tbl += "\n".join(_emg_ts_lines)

    _fn = " (bandpass + RMS)" if emg_task_filter_switch.value else ""
    _bl = " with baseline correction" if emg_baseline_switch.value else ""

    mo.md(f"""
    ### EMG Per-Task Summary{_fn}{_bl}

    Mean and max EMG (mV) of the global sensor average, from the condition-level averaged curve (matching the graph).

    {_emg_ts_tbl}
    """)
    return


@app.cell(hide_code=True)
def _(
    Line2D,
    apply_baseline,
    build_legend,
    emg_baseline_switch,
    emg_df,
    emg_task_filter_switch,
    emg_task_selector,
    legend_layout,
    pl,
    plt,
    prepare_emg_for_analysis,
):
    # EMG Per-Task Interactive Plot
    # Reuses: emg_df, pl, plt, np, Line2D, emg_task_selector, filter_rms, emg_task_filter_switch, apply_baseline

    _emg_src = prepare_emg_for_analysis(
        emg_df,
        apply_filter=emg_task_filter_switch.value,
        time_min=-5.0,
        time_max=15.0,
    )
    _ylabel = "EMG RMS (mV)" if emg_task_filter_switch.value else "EMG (mV)"

    _emg_tp_TIME_MIN = -5.0
    _emg_tp_TIME_MAX = 15.0

    _emg_tp_filtered = _emg_src.filter(
        (pl.col("time_sec") >= _emg_tp_TIME_MIN)
        & (pl.col("time_sec") <= _emg_tp_TIME_MAX)
    )

    _emg_tp_ec = [c for c in _emg_src.columns if "EMG" in c and c.endswith("(mV)")]

    _emg_tp_selected = emg_task_selector.value

    if not _emg_tp_selected:
        _emg_tp_fig, _ax = plt.subplots(figsize=(14, 5))
        _ax.text(
            0.5, 0.5, "No tasks selected", ha="center", va="center", fontsize=14
        )
        _ax.set_axis_off()

    else:
        _emg_tp_robot = {}
        _emg_tp_no_robot = {}

        for _task in _emg_tp_selected:
            _task_df = _emg_tp_filtered.filter(pl.col("task") == _task)

            _task_run_avg = _task_df.with_columns(
                pl.mean_horizontal(_emg_tp_ec).alias("emg_mean")
            )

            # Baseline correction
            if emg_baseline_switch.value:
                _task_run_avg = apply_baseline(_task_run_avg, ["emg_mean"])

            _task_avg = (
                _task_run_avg.group_by("time_sec", "is_robot")
                .agg(pl.col("emg_mean").mean().alias("emg"))
                .sort("time_sec")
            )

            _emg_tp_robot[_task] = _task_avg.filter(pl.col("is_robot") == True)
            _emg_tp_no_robot[_task] = _task_avg.filter(pl.col("is_robot") == False)

        _n_tasks = len(_emg_tp_selected)
        _cmap = plt.cm.tab10 if _n_tasks <= 10 else plt.cm.tab20
        _emg_tp_colors = {
            _t: _cmap(_i / max(_n_tasks - 1, 1))
            for _i, _t in enumerate(_emg_tp_selected)
        }

        _emg_tp_fig, (_emg_tp_ax1, _emg_tp_ax2) = plt.subplots(
            1, 2, figsize=(14, 5)
        )

        for _task in _emg_tp_selected:
            _c = _emg_tp_colors[_task]
            _emg_tp_ax1.plot(
                _emg_tp_robot[_task]["time_sec"].to_numpy(),
                _emg_tp_robot[_task]["emg"].to_numpy(),
                color=_c,
                label=_task,
                linewidth=0.8,
            )
            _emg_tp_ax2.plot(
                _emg_tp_no_robot[_task]["time_sec"].to_numpy(),
                _emg_tp_no_robot[_task]["emg"].to_numpy(),
                color=_c,
                label=_task,
                linewidth=0.8,
            )

        _emg_tp_ax1.set_title("Robot Trials")
        _emg_tp_ax1.set_xlabel("Time (s)")
        _emg_tp_ax1.set_ylabel(_ylabel)
        _emg_tp_ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        _emg_tp_ax2.set_title("No-Robot Trials")
        _emg_tp_ax2.set_xlabel("Time (s)")
        _emg_tp_ax2.set_ylabel(_ylabel)
        _emg_tp_ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        _legend_handles = []
        for _task in _emg_tp_selected:
            _c = _emg_tp_colors[_task]
            _legend_handles.append(Line2D([0], [0], color=_c, label=_task))

        legend_layout(_emg_tp_fig, len(_legend_handles))
        build_legend(_emg_tp_fig, _legend_handles)



    _emg_tp_fig
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

    pr_run_select = mo.ui.dropdown(
        options=sorted(fnirs_df["run_id"].unique().to_list()), label="Run"
    )

    pr_baseline_switch = mo.ui.switch(label="Baseline correction (-5 to 0s)")
    pr_filter_switch = mo.ui.switch(
        label="Bandpass + RMS filter (20-450 Hz, 100 ms)"
    )

    mo.md("### Per-Run Viewer")
    mo.vstack(
        [
            pr_task_select,
            pr_run_select,
            pr_baseline_switch,
            pr_filter_switch,
        ]
    )
    return pr_baseline_switch, pr_filter_switch, pr_run_select, pr_task_select


@app.cell(hide_code=True)
def _(
    emg_df,
    fnirs_df,
    mo,
    pl,
    pr_filter_switch,
    pr_run_select,
    pr_task_select,
    prepare_emg_for_analysis,
):
    # Per-Run Viewer Summary Stats
    # Reuses: fnirs_df, emg_df, pr_task_select, pr_robot_select, pr_run_select, pr_baseline_switch, pr_filter_switch, filter_rms

    _lines = []

    if pr_run_select.value is None or pr_task_select.value is None:
        _lines.append("*Select a run and task to see summary statistics.*")
    else:
        _cond = (
            (pl.col("run_id") == pr_run_select.value)
            & (pl.col("task") == pr_task_select.value)
        )
        _cond_post = _cond & (pl.col("time_sec") >= 0)

        _r_fnirs = fnirs_df.filter(_cond)

        # Apply filter to EMG if toggle is on
        _emg_for_stats = prepare_emg_for_analysis(
        emg_df,
        apply_filter=pr_filter_switch.value,
        time_min=-5.0,
        time_max=15.0,
    )
        _r_emg = _emg_for_stats.filter(_cond)
        _r_emg_post = _emg_for_stats.filter(_cond_post)

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
            f"**{_participant}** \u2014 {pr_run_select.value} \u2014 {pr_task_select.value} \u2014"
        )
        _lines.append("")

        # fNIRS stats
        _pr_hbo_cols = [c for c in fnirs_df.columns if c.endswith("_hbo")]
        _pr_hbr_cols = [c for c in fnirs_df.columns if c.endswith("_hbr")]
        _r_fnirs_post = fnirs_df.filter(_cond_post)

        _fnirs_n = 0
        _hbo_mean = _hbo_peak = _hbr_mean = _hbr_peak = 0.0
        if _r_fnirs_post.height > 0 and len(_pr_hbo_cols) > 0:
            _hbo_vals = _r_fnirs_post.select(
                pl.mean_horizontal(_pr_hbo_cols)
            ).drop_nulls()
            _hbr_vals = _r_fnirs_post.select(
                pl.mean_horizontal(_pr_hbr_cols)
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
        _pr_emg_cols = [
            c for c in _emg_for_stats.columns if "EMG" in c and c.endswith("(mV)")
        ]
        _emg_n = 0
        if _r_emg_post.height > 0 and len(_pr_emg_cols) > 0:
            _emg_n = _r_emg_post.height

        _fn = " (filtered)" if pr_filter_switch.value else ""
        _lines.append(
            f"**EMG{_fn}** ({_emg_freq}{', ' + f'{_emg_n:,}' + ' post-stimulus samples' if _emg_n > 0 else ''})"
        )
        _lines.append("")
        _lines.append("| Sensor | Mean (mV) | Peak (mV) |")
        _lines.append("|--------|-----------|-----------|")
        if _emg_n > 0:
            for _col in _pr_emg_cols:
                _parts = _col.split(" | ")
                _sensor_name = _parts[0]
                _emg_ch = (
                    _parts[1].replace(" (mV)", "") if len(_parts) > 1 else _col
                )
                _lbl = f"{_sensor_name} {_emg_ch}"
                _vals = _r_emg_post[_col].drop_nulls()
                if len(_vals) > 0:
                    _m = _vals.mean()
                    _p = _vals.max()
                    _lines.append(f"| {_lbl} | {_m:.4f} | {_p:.4f} |")
                else:
                    _lines.append(f"| {_lbl} | \u2014 | \u2014 |")
        else:
            for _col in _pr_emg_cols:
                _parts = _col.split(" | ")
                _sensor_name = _parts[0]
                _emg_ch = (
                    _parts[1].replace(" (mV)", "") if len(_parts) > 1 else _col
                )
                _lbl = f"{_sensor_name} {_emg_ch}"
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
    pr_filter_switch,
    pr_run_select,
    pr_task_select,
    prepare_emg_for_analysis,
):
    # Per-Run Viewer Plot
    # Reuses: fnirs_df, emg_df, pl, plt, np, pr_task_select, pr_robot_select, pr_run_select, pr_baseline_switch, pr_filter_switch, filter_rms

    _pr_TIME_MIN = -5.0
    _pr_TIME_MAX = 15.0

    _pr_hbo_cols = [c for c in fnirs_df.columns if c.endswith("_hbo")]
    _pr_hbr_cols = [c for c in fnirs_df.columns if c.endswith("_hbr")]

    # Use filtered EMG if toggle is on
    _pr_emg_for_plot = prepare_emg_for_analysis(
        emg_df,
        apply_filter=pr_filter_switch.value,
        time_min=-5.0,
        time_max=15.0,
    )
    _pr_emg_cols = [
        c for c in _pr_emg_for_plot.columns if "EMG" in c and c.endswith("(mV)")
    ]

    # EMG labels
    _pr_emg_labels = []
    for _c in _pr_emg_cols:
        _parts = _c.split(" | ")
        _sensor_name = _parts[0]
        _emg_ch = _parts[1].replace(" (mV)", "") if len(_parts) > 1 else _c
        _pr_emg_labels.append(f"{_sensor_name} {_emg_ch}")

    _ylabel_emg = "EMG RMS (mV)" if pr_filter_switch.value else "EMG (mV)"

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


    else:
        _cond_filter = (
            (pl.col("run_id") == pr_run_select.value)
            & (pl.col("task") == pr_task_select.value)
            & (pl.col("time_sec") >= _pr_TIME_MIN)
            & (pl.col("time_sec") <= _pr_TIME_MAX)
        )

        _run_fnirs = fnirs_df.filter(_cond_filter).sort("time_sec")
        _run_emg = _pr_emg_for_plot.filter(_cond_filter).sort("time_sec")

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


        else:
            # ============================================================
            # fNIRS
            # ============================================================
            if (
                len(_pr_hbo_cols) == 0
                or len(_pr_hbr_cols) == 0
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
                            pl.mean_horizontal(_pr_hbo_cols).alias("HbO"),
                            pl.mean_horizontal(_pr_hbr_cols).alias("HbR"),
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
            for _col in _pr_emg_cols:
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
            _n_emg = len(_pr_emg_cols)
            pr_fig, _axes = plt.subplots(
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
                _axes[0].set_ylabel("Concentration (\u03bcM)")
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
                f"fNIRS \u2014 {pr_run_select.value} \u2014 {pr_task_select.value} \u2014"
            )

            # EMG plots
            for i, (_col, _label) in enumerate(zip(_pr_emg_cols, _pr_emg_labels)):
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
                _ax.set_xlim(_pr_TIME_MIN, _pr_TIME_MAX)

            plt.tight_layout()

    pr_fig
    return


@app.cell
def _(emg_df, pl):
    emg_sampling_check = (
        emg_df
        .sort(["run_id", "task_instance", "time_sec"])
        .with_columns(
            pl.col("time_sec")
            .diff()
            .over(["run_id", "task_instance"])
            .alias("dt")
        )
        .group_by(["run_id", "task_instance"])
        .agg(
            [
                pl.len().alias("n_samples"),
                pl.col("time_sec").min().alias("t_min"),
                pl.col("time_sec").max().alias("t_max"),
                pl.col("dt").drop_nulls().min().alias("dt_min"),
                pl.col("dt").drop_nulls().median().alias("dt_median"),
                pl.col("dt").drop_nulls().max().alias("dt_max"),
                pl.col("dt").drop_nulls().mean().alias("dt_mean"),
                pl.col("dt").drop_nulls().std().alias("dt_std"),
                (1 / pl.col("dt").drop_nulls().median()).alias("fs_median"),
                (pl.col("dt") <= 0).sum().alias("non_positive_dt_count"),
            ]
        )
        .sort(["run_id", "task_instance"])
    )

    emg_sampling_check
    return


if __name__ == "__main__":
    app.run()
