import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


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
    from collections import defaultdict
    import pandas as pd

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
    sys.path.insert(0, "/Users/haider/code/Franka-fNIRS-EMG-data_analysis")
    from src.loaders.loader import load_data

    data_files = load_data(RAW_DIR)
    print(data_files)

    mo.md(f"**Found {len(data_files)} participants:** `{list(data_files.keys())}`")
    return (data_files,)


@app.cell
def _(data_files, mne):
    def _():
        """Print per-session epoch statistics: total, kept, and dropped counts."""
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
        return print(f"{'NOROBOT TOTAL':<22} {'':<10} {norobot_total:>7} {norobot_kept:>7} {norobot_total-norobot_kept:>8}")


    _()
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
    fig, axes = plt.subplots(ncols=2, figsize=(14, 5), layout="constrained", sharey=True)
    axes[0].set_title("Robot — HbO & HbR")
    axes[1].set_title("NoRobot — HbO & HbR")

    colors = {"hbo": "#AA3377", "hbr": "#4477AA"}
    for ax, _label, _epochs in zip(
        axes, ["Robot", "NoRobot"], [robot_epochs, norobot_epochs]
    ):
        for chrom in ["hbo", "hbr"]:
            ev = _epochs.average(picks=chrom)
            ax.plot(
                ev.times,
                ev.data.mean(axis=0) * 1e6,
                label=chrom.upper(),
                color=colors[chrom],
            )
        ax.set(xlabel="Time (s)", ylabel="Concentration (μM)")
        ax.legend()
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    plt.savefig("./figures/combined_hbo-hbr.png")
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
    plt.savefig("./figures/per-task_hbo-hbr.png")
    plt.show()
    return


@app.cell
def _(norobot_epochs, os, plt, robot_epochs):
    def _():
        cmp_tasks = [f"task_{i}" for i in range(1, 11)]

        os.makedirs("./figures", exist_ok=True)

        cmp_colors = {"Robot": "#AA3377", "NoRobot": "#4477AA"}

        for cmp_task in cmp_tasks:
            fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")

            for label, cmp_epochs in [
                ("Robot", robot_epochs),
                ("NoRobot", norobot_epochs),
            ]:
                for ch, ls in [("hbo", "-"), ("hbr", "--")]:
                    cmp_ev = cmp_epochs[cmp_task].average(picks=ch)

                    ax.plot(
                        cmp_ev.times,
                        cmp_ev.data.mean(axis=0) * 1e6,
                        label=f"{label} {ch.upper()}",
                        color=cmp_colors[label],
                        linestyle=ls,
                    )

            ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
            ax.legend(fontsize=8)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Concentration (μM)")
            ax.tick_params(labelsize=8)

            # No title

            filename = f"./figures/{cmp_task}_hbo-hbr_robot-vs-norobot.png"
            fig.savefig(filename, dpi=300, bbox_inches="tight")
        return plt.close(fig)


    _()
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


@app.cell(disabled=True)
def _(PROCESSED_DIR, data_files, mne, os, pl, preprocess, task_ids):
    # Export per-session fNIRS epochs to a single long-format CSV.
    # Each row = one time sample within one epoch, with all HbO/HbR channel
    # values and metadata (participant, task, condition, etc.).

    def parse_dataset_key(dataset_key):
        """Extract participant name, run_id, and robot condition from a dataset key.

        Parses keys like "sam_robot_1" or "clarence_norobot" into structured
        components. The run_id is reformatted to "{participant}_{condition}_{num}".

        Args:
            dataset_key: Raw folder key from load_data() (e.g. "sam_robot_1").

        Returns:
            Tuple of (participant, run_id, is_robot).
        """
        is_robot = "_robot_" in dataset_key and "_norobot_" not in dataset_key

        if "_norobot_" in dataset_key:
            participant, run_str = dataset_key.rsplit("_norobot_", 1)
            condition = "norobot"
        elif "_robot_" in dataset_key:
            participant, run_str = dataset_key.rsplit("_robot_", 1)
            condition = "robot"
        else:
            participant = dataset_key
            run_str = None
            condition = "unknown"

        try:
            run_num = int(run_str) if run_str is not None else None
        except ValueError:
            run_num = run_str

        # 🔹 Build new run_id string
        if run_num is not None:
            run_id = f"{participant}_{condition}_{run_num}"
        else:
            run_id = f"{participant}_{condition}"

        return participant, run_id, is_robot


    def build_event_metadata(events, dataset_key, task_id_map):
        """Build a Polars DataFrame of event metadata from MNE event arrays.

        Each row represents one task event with its dataset origin, participant,
        condition, task label, and sample-level timing info.

        Args:
            events:      MNE events array (N×3: sample, duration, event_code).
            dataset_key: Source session key (e.g. "sam_robot_1").
            task_id_map: Dict mapping task names to event codes.

        Returns:
            Polars DataFrame with columns: dataset_key, participant, run_id,
            is_robot, task, task_instance, epoch_index_in_dataset,
            event_sample, event_code.
        """
        participant, run_id, is_robot = parse_dataset_key(dataset_key)
        id_to_task = {v: k for k, v in task_id_map.items()}

        rows = []
        for i, event in enumerate(events):
            event_sample = int(event[0])
            event_code = int(event[2])
            task = id_to_task.get(event_code, f"unknown_{event_code}")

            rows.append(
                {
                    "dataset_key": dataset_key,
                    "participant": participant,
                    "run_id": run_id,
                    "is_robot": is_robot,
                    "task": task,
                    "task_instance": i + 1,
                    "epoch_index_in_dataset": i,
                    "event_sample": event_sample,
                    "event_code": event_code,
                }
            )
        return pl.DataFrame(rows)


    def make_epochs_for_dataset(dataset_key, fnirs_path, preprocess, task_ids):
        """Load one SNiRF file, preprocess, and extract task epochs.

        Applies the full fNIRS pipeline (optical density → Beer-Lambert →
        bad channel detection → bandpass filter), then creates epochs
        time-locked to task events with artifact rejection.

        Args:
            dataset_key: Session key (e.g. "sam_robot_1").
            fnirs_path:  Path to the .snirf file.
            preprocess:  Preprocessing function (see preprocess() above).
            task_ids:    Dict mapping task names to event codes.

        Returns:
            MNE Epochs object with metadata attached.
        """
        raw = mne.io.read_raw_snirf(fnirs_path, preload=True, verbose=False)
        raw.resample(10.0)

        _, _, raw_hb, _ = preprocess(raw, label=dataset_key)

        events, _ = mne.events_from_annotations(raw_hb, verbose=False)
        metadata = build_event_metadata(events, dataset_key, task_ids).to_pandas()

        epochs = mne.Epochs(
            raw_hb,
            events,
            event_id=task_ids,
            tmin=-5,
            tmax=15,
            baseline=(None, 0),
            reject=dict(hbo=100e-6, hbr=100e-6),
            preload=True,
            metadata=metadata,
            verbose=False,
        )

        n_before = len(epochs)
        epochs.drop_bad()
        n_after = len(epochs)
        print(f"{dataset_key}: kept {n_after} / {n_before} epochs")

        return epochs


    def epochs_to_long(epochs):
        """Reshape MNE Epochs from 3D (epochs × channels × times) to long format.

        Each row is one time sample within one epoch, with all channel values
        as columns. Metadata columns (participant, task, etc.) are duplicated
        across time points for each epoch.

        Args:
            epochs: MNE Epochs object with attached metadata.

        Returns:
            Polars DataFrame in long format with time_sec, channel columns,
            and all metadata columns.
        """
        data = epochs.get_data()
        times = epochs.times
        ch_names = epochs.ch_names

        meta_df = pl.from_pandas(epochs.metadata.reset_index(drop=True))

        rows = []
        for epoch_idx in range(data.shape[0]):
            meta = meta_df.row(epoch_idx, named=True)

            for time_idx, time_sec in enumerate(times):
                row = dict(meta)
                row["time_index"] = time_idx
                row["time_sec"] = float(time_sec)

                for ch_idx, ch in enumerate(ch_names):
                    safe_ch = ch.replace(" ", "_")
                    row[safe_ch] = float(data[epoch_idx, ch_idx, time_idx])

                rows.append(row)

        return pl.DataFrame(rows)



    long_dfs = []
    dataset_summary = []

    for dataset_key, paths in sorted(data_files.items()):
        epochs = make_epochs_for_dataset(
            dataset_key=dataset_key,
            fnirs_path=paths["fNIRS"],
            preprocess=preprocess,
            task_ids=task_ids,
        )

        long_df_dataset = epochs_to_long(epochs)
        long_dfs.append(long_df_dataset)

        md = epochs.metadata
        dataset_summary.append(
            {
                "dataset_key": dataset_key,
                "participant": md["participant"].iloc[0] if len(md) else None,
                "run_id": md["run_id"].iloc[0] if len(md) else None,
                "is_robot": md["is_robot"].iloc[0] if len(md) else None,
                "kept_epochs": len(epochs),
            }
        )

    long_df = pl.concat(long_dfs, how="diagonal")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    export_path = f"{PROCESSED_DIR}/combined_fnirs.csv"
    long_df.write_csv(export_path)

    meta_cols = [
        "dataset_key",
        "participant",
        "run_id",
        "is_robot",
        "task",
        "task_instance",
        "epoch_index_in_dataset",
        "event_sample",
        "event_code",
    ]

    print(f"Exported: {export_path}")
    print(f"Rows: {long_df.height}")
    print(f"Columns: {long_df.width}")
    print("Metadata columns:", meta_cols)
    print(pl.DataFrame(dataset_summary))
    print(long_df.select(meta_cols).head(10))
    return (long_df,)


@app.cell
def _(long_df):
    long_df
    return


if __name__ == "__main__":
    app.run()
