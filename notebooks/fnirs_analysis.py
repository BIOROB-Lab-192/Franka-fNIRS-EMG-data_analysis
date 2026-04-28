import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import mne
    import numpy as np
    import polars as pl
    import os
    from itertools import compress
    import re

    DATA_DIR = "data"
    RAW_DIR = f"{DATA_DIR}/raw"
    PROCESSED_DIR = f"{DATA_DIR}/processed"
    FIGURES_DIR = "../figures"

    mo.md(f"""
    ### Project paths
    - **Raw data:** `{RAW_DIR}`
    - **Processed output:** `{PROCESSED_DIR}`
    - **Figures:** `{FIGURES_DIR}`
    """)
    return PROCESSED_DIR, RAW_DIR, mne, mo, np, os, pl, plt


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
def _(data_files, mne):
    session_stats = {}
    for key, paths in data_files.items():
        raw = mne.io.read_raw_snirf(paths["fNIRS"], preload=True, verbose=False)
        raw.load_data().resample(10.0)
        raw_fnirs = raw.copy().pick("fnirs")
        raw_od = mne.preprocessing.nirs.optical_density(raw_fnirs)
        raw_hb = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6.0)
        events, _ = mne.events_from_annotations(raw_hb, verbose=False)
        _task_ids = {
            'task_1': 1, 'task_2': 2, 'task_3': 3, 'task_4': 4, 'task_5': 5,
            'task_6': 6, 'task_7': 7, 'task_8': 8, 'task_9': 9, 'task_10': 10,
        }
        epochs = mne.Epochs(raw_hb, events, _task_ids, tmin=-5, tmax=15,
                             baseline=(None, 0), preload=True)
        epochs.drop_bad()
        n_total = len(events)
        n_kept = len(epochs)
        n_dropped = n_total - n_kept
        condition = "robot" if "_robot" in key and "norobot" not in key else "norobot"
        session_stats[key] = dict(condition=condition, total=n_total,
                                   kept=n_kept, dropped=n_dropped)

    print(f"{'Session':<22} {'Condition':<10} {'Total':>7} {'Kept':>7} {'Dropped':>8}")
    print("-" * 58)
    for k, v in sorted(session_stats.items()):
        print(f"{k:<22} {v['condition']:<10} {v['total']:>7} {v['kept']:>7} {v['dropped']:>8}")
    print("-" * 58)
    robot_total = sum(v["total"] for v in session_stats.values() if v["condition"] == "robot")
    norobot_total = sum(v["total"] for v in session_stats.values() if v["condition"] == "norobot")
    robot_kept = sum(v["kept"] for v in session_stats.values() if v["condition"] == "robot")
    norobot_kept = sum(v["kept"] for v in session_stats.values() if v["condition"] == "norobot")
    print(f"{'ROBOT TOTAL':<22} {'':<10} {robot_total:>7} {robot_kept:>7} {robot_total-robot_kept:>8}")
    print(f"{'NOROBOT TOTAL':<22} {'':<10} {norobot_total:>7} {norobot_kept:>7} {norobot_total-norobot_kept:>8}")
    return


@app.cell
def _(data_files, mne):
    robot_paths = [v["fNIRS"] for k, v in data_files.items() if "_robot" in k and "norobot" not in k]
    norobot_paths = [v["fNIRS"] for k, v in data_files.items() if "norobot" in k]

    robot_raws = [mne.io.read_raw_snirf(p, preload=True, verbose=False) for p in robot_paths]
    norobot_raws = [mne.io.read_raw_snirf(p, preload=True, verbose=False) for p in norobot_paths]

    # Resample to 10.0 Hz — fixes floating-point drift between sessions
    target_sfreq = 10.0
    robot_raws = [r.resample(target_sfreq) for r in robot_raws]
    norobot_raws = [r.resample(target_sfreq) for r in norobot_raws]

    robot_raw = mne.concatenate_raws(robot_raws)
    norobot_raw = mne.concatenate_raws(norobot_raws)

    robot_raw.load_data()
    norobot_raw.load_data()

    task_ids = {
        'task_1': 1, 'task_2': 2, 'task_3': 3, 'task_4': 4, 'task_5': 5,
        'task_6': 6, 'task_7': 7, 'task_8': 8, 'task_9': 9, 'task_10': 10,
    }
    return norobot_raw, robot_raw, task_ids


@app.cell
def _(mne):
    def preprocess(raw, label=""):
        """Run the full preprocessing pipeline on a concatenated raw."""
        raw_fnirs = raw.copy().pick("fnirs")
        raw_od = mne.preprocessing.nirs.optical_density(raw_fnirs)

        # Scalp coupling index — mark bad channels
        sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
        from itertools import compress
        raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.3))
        n_bads = len(raw_od.info["bads"])

        # Beer-Lambert law
        raw_hb = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6.0)

        # Bandpass filter
        raw_haemo_unfiltered = raw_hb.copy()
        raw_hb.filter(l_freq=0.01, h_freq=0.7, h_trans_bandwidth=0.1)

        print(f"[{label}] Bad channels: {n_bads} / {len(raw_od.ch_names)}")
        return raw_fnirs, raw_od, raw_hb, raw_haemo_unfiltered

    return (preprocess,)


@app.cell
def _(mo, norobot_raw, preprocess):
    norobot_fnirs, norobot_od, norobot_hb, norobot_hb_unfilt = preprocess(norobot_raw, label="NoRobot")

    mo.md(f"**NoRobot pipeline done** — {norobot_hb.times[-1]:.0f}s of data")
    return (norobot_hb,)


@app.cell
def _(mo, preprocess, robot_raw):
    robot_fnirs, robot_od, robot_hb, robot_hb_unfilt = preprocess(robot_raw, label="Robot")

    mo.md(f"**Robot pipeline done** — {robot_hb.times[-1]:.0f}s of data")
    return (robot_hb,)


@app.cell
def _(mne, mo, norobot_hb):
    norobot_events, norobot_id = mne.events_from_annotations(norobot_hb, verbose=False)
    mo.md(f"**Norobot events:** {len(norobot_events)} found")
    return (norobot_events,)


@app.cell
def _(mne, mo, robot_hb):
    robot_events, robot_id = mne.events_from_annotations(robot_hb, verbose=False)
    mo.md(f"**Robot events:** {len(robot_events)} found")
    return (robot_events,)


@app.cell
def _(mne, mo, norobot_events, norobot_hb, task_ids):
    norobot_reject = dict(hbo=100e-6, hbr=100e-6)
    norobot_epochs = mne.Epochs(
        norobot_hb, norobot_events, task_ids,
        tmin=-5, tmax=15, baseline=(None, 0),
        reject=norobot_reject, preload=True,
    )
    norobot_epochs.drop_bad()
    mo.md(f"**Norobot epochs:** {len(norobot_epochs)} / {len(norobot_events)}")
    return (norobot_epochs,)


@app.cell
def _(mne, mo, robot_events, robot_hb, task_ids):
    robot_reject = dict(hbo=100e-6, hbr=100e-6)
    robot_epochs = mne.Epochs(
        robot_hb, robot_events, task_ids,
        tmin=-5, tmax=15, baseline=(None, 0),
        reject=robot_reject, preload=True,
    )
    robot_epochs.drop_bad()
    mo.md(f"**Robot epochs:** {len(robot_epochs)} / {len(robot_events)}")
    return (robot_epochs,)


@app.cell
def _(norobot_epochs, plt, robot_epochs):
    fig, axes = plt.subplots(ncols=2, figsize=(14, 5), layout="constrained")
    axes[0].set_title("Robot vs NoRobot — HbO")
    axes[1].set_title("Robot vs NoRobot — HbR")

    colors = {"Robot": "#AA3377", "NoRobot": "#4477AA"}
    for ax, chrom in zip(axes, ["hbo", "hbr"]):
        for _label, _epochs in [
            ("Robot", robot_epochs),
            ("NoRobot", norobot_epochs),
        ]:
            ev = _epochs.average(picks=chrom)
            ax.plot(
                ev.times,
                ev.data.mean(axis=0) * 1e6,
                label=_label,
                color=colors[_label],
            )
        ax.set(xlabel="Time (s)", ylabel=f"{chrom.upper()} (μM)")
        ax.legend()
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    plt.show()
    return


@app.cell
def _(norobot_epochs, plt, robot_epochs):
    cmp_tasks = [f"task_{i}" for i in range(1, 11)]
    cmpfig, cmpaxes = plt.subplots(
        nrows=2, ncols=5, figsize=(20, 8), layout="constrained"
    )
    cmp_axes_flat = cmpaxes.ravel()

    cmp_colors = {"Robot": "#AA3377", "NoRobot": "#4477AA"}
    for cmp_ax, cmp_task in zip(cmp_axes_flat, cmp_tasks):
        for label, cmp_epochs in [
            ("Robot", robot_epochs),
            ("NoRobot", norobot_epochs),
        ]:
            for ch, ls in [("hbo", "-"), ("hbr", "--")]:
                cmp_ev = cmp_epochs[cmp_task].average(picks=ch)
                cmp_ax.plot(
                    cmp_ev.times,
                    cmp_ev.data.mean(axis=0) * 1e6,
                    label=f"{label} {ch.upper()}",
                    color=cmp_colors[label],
                    linestyle=ls,
                )
        cmp_ax.set_title(cmp_task, fontsize=10)
        cmp_ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
        cmp_ax.legend(fontsize=6)
        cmp_ax.set_xlabel("Time (s)", fontsize=7)
        cmp_ax.set_ylabel("Concentration (μM)", fontsize=7)
        cmp_ax.tick_params(labelsize=7)
    plt.show()
    return


@app.cell
def _(norobot_epochs, np, plt, robot_epochs):
    jnt_tasks = [f"task_{i}" for i in range(1, 11)]
    jnt_times = np.arange(-1.0, 12.0, 2.5)  # key timepoints for topomaps

    for jnt_task in jnt_tasks:
        jnt_fig = norobot_epochs[jnt_task].average(picks="hbo").plot_joint(
            times=jnt_times, topomap_args=dict(extrapolate="local")
        )
        jnt_fig.suptitle(f"NoRobot — {jnt_task} — HbO", fontsize=12)

        jnt_fig2 = robot_epochs[jnt_task].average(picks="hbo").plot_joint(
            times=jnt_times, topomap_args=dict(extrapolate="local")
        )
        jnt_fig2.suptitle(f"Robot — {jnt_task} — HbO", fontsize=12)

    plt.show()
    return


@app.cell
def _(PROCESSED_DIR, mo, norobot_epochs, os, pl, robot_epochs, task_ids):
    def epochs_to_wide(epochs, condition, task_id_map):
        """
        Build a wide Polars DataFrame from an MNE Epochs object.
        One row per epoch; columns = condition, task, is_bad, then
        {HbO/HbR}_{ch_name}_t{0..n_times-1} for all channels.
        Includes ALL epochs — kept and bad — so no data is silently dropped.
        """
        ch_names = epochs.ch_names
        n_epochs, n_channels, n_times = epochs.get_data().shape

        # Reverse task_id map: event_id → "task_N"
        id_to_task = {v: k for k, v in task_id_map.items()}

        # Drop log: True = dropped, False = kept
        drop_log = epochs.drop_log

        rows = []
        for i in range(n_epochs):
            event_id = epochs.events[i, 2]
            task = id_to_task.get(event_id, f"unknown_{event_id}")
            is_bad = drop_log[i] is not None and len(drop_log[i]) > 0

            # (n_channels, n_times) — baseline-corrected values
            epoch_data = epochs.get_data()[i]

            row = dict(
                condition=condition,
                task=task,
                is_bad=is_bad,
            )
            for ch_idx, ch in enumerate(ch_names):
                # prepend channel type for disambiguation
                if ch.startswith("HbO"):
                    prefix = "HbO"
                elif ch.startswith("HbR"):
                    prefix = "HbR"
                else:
                    prefix = "AUX"
                for t in range(n_times):
                    row[f"{prefix}_{ch}_t{t}"] = epoch_data[ch_idx, t]
            rows.append(row)

        return pl.DataFrame(rows)

    # Build wide tables for both conditions
    norobot_df = epochs_to_wide(norobot_epochs, "norobot", task_ids)
    robot_df = epochs_to_wide(robot_epochs, "robot", task_ids)

    # Combine
    wide_df = pl.concat([norobot_df, robot_df])

    _n_total = len(wide_df)
    n_bad = wide_df.filter(pl.col("is_bad") == True).height
    n_good = _n_total - n_bad
    n_cols = wide_df.width

    # Export to CSV
    EXPORT_DIR = PROCESSED_DIR
    os.makedirs(EXPORT_DIR, exist_ok=True)
    export_path = f"{EXPORT_DIR}/epochs_wide.csv"
    wide_df.write_csv(export_path)

    mo.md(f"""
    **Export done** — `{export_path}`
    - Total epochs: `{_n_total}` (`{n_bad}` bad / `{n_good}` good)
    - Columns: `{n_cols}` (condition, task, is_bad + `{n_cols - 3}` channel×time features)
    - Tasks: `{sorted(task_ids.keys())}`
    """)
    return


if __name__ == "__main__":
    app.run()
