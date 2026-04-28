import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import mne
    import os
    from itertools import compress
    import numpy as np
    import re

    DATA_DIR = "data"
    RAW_DIR = f"{DATA_DIR}/raw"
    PROCESSED_DIR = f"{DATA_DIR}/processed"
    FIGURES_DIR = "../figures"

    SNIRF_FILE = f"{RAW_DIR}/sam_robot1/2026-04-24_003/2026-04-24_003.snirf"

    mo.md(f"""
    ### Project paths
    - **Raw data:** `{RAW_DIR}`
    - **Processed output:** `{PROCESSED_DIR}`
    - **Figures:** `{FIGURES_DIR}`
    - **Current file:** `{SNIRF_FILE}`
    """)
    return RAW_DIR, compress, mne, mo, np, plt


@app.cell
def _(RAW_DIR, mo):
    import sys
    sys.path.insert(0, "/Users/haider/code/data_analysis")
    from src.loaders.loader import load_data

    data_files = load_data(RAW_DIR)

    mo.md(f"**Found {len(data_files)} participants:** `{list(data_files.keys())}`")
    return (data_files,)


@app.cell
def _(data_files, mne, mo):
    PATH = data_files["clarence_norobot_1"]["fNIRS"]

    raw = mne.io.read_raw_snirf(PATH, preload=True, verbose=False)
    raw.load_data()

    mo.md(f"""
    **Loaded:** `{PATH}`
    - **Channels:** {len(raw.ch_names)}
    - **Duration:** {raw.times[-1]:.1f}s @ {raw.info['sfreq']:.1f} Hz
    """)
    return (raw,)


@app.cell
def _(raw):
    print(raw.info)
    return


@app.cell
def _(mne, mo, raw):
    # Pick only fNIRS channels (drop auxiliary channels)
    raw_fnirs = raw.copy().pick("fnirs")

    # Convert to optical density
    raw_od = mne.preprocessing.nirs.optical_density(raw_fnirs)

    # Apply Beer-Lambert law → HbO and HbR concentrations
    # ppf: partial pathlength factor (default 6.0 for adult cortex)
    raw_hb = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6.0)
    raw_haemo_unfiltered = raw_hb.copy()

    mo.md(f"**Converted to HbO/HbR** — {len(raw_hb.ch_names)} hemoglobin channels")
    return raw_haemo_unfiltered, raw_hb, raw_od


@app.cell
def _(compress, mne, mo, plt, raw_od):
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    sci_fig, sci_ax = plt.subplots(layout="constrained")
    sci_ax.hist(sci)
    sci_ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])

    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.2))

    plt.show()

    mo.md(f"**Bad channel total** {len(raw_od.info["bads"])}")
    return


@app.cell
def _(plt, raw_haemo_unfiltered, raw_hb):
    # Bandpass: 0.01–0.7 Hz (removes slow drift and cardiac noise)
    raw_hb.filter(l_freq=0.01, h_freq=0.7, h_trans_bandwidth=0.1)

    for when, _raw in dict(Before=raw_haemo_unfiltered, After=raw_hb).items():
        psd_fig = _raw.compute_psd().plot(
            average=True, amplitude=False, picks="data", exclude="bads"
        )
        psd_fig.suptitle(f"{when} filtering", weight="bold", size="x-large")
    plt.show()
    # mo.md("**Filtered:** 0.01 – 0.5 Hz bandpass")
    return


@app.cell
def _(mne, mo, raw_hb):
    # fNIRS events from SNiRF annotations
    events, event_id = mne.events_from_annotations(raw_hb, verbose=False)

    # Map numeric event IDs to condition names
    # Edit these labels to match your experiment protocol
    # Note: event_id keys are numpy strings, so we convert with int()


    task_ids = {
        'task_1': 1,
        'task_2': 2,
        'task_3': 3,
        'task_4': 4,
        'task_5': 5,
        'task_6': 6,
        'task_7': 7,
        'task_8': 8,
        'task_9': 9,
        'task_10': 10,
    }


    epochs_fig = mne.viz.plot_events(events, event_id=task_ids, sfreq=raw_hb.info["sfreq"])

    mo.md(f"""
    **Found {len(events)} events**
    **Event IDs:** {task_ids}
    """)
    return events, task_ids


@app.cell
def _(events, mne, mo, plt, raw_hb, task_ids):
    # tmin/tmax: time window around each event (in seconds)
    # baseline: baseline correction period (None = use pre-event)
    # reject: channel-type-level amplitude thresholds for artifact rejection
    #   Note: MNE uses lowercase 'hbo'/'hbr' as channel type names

    reject_criteria = dict(
        hbo=100e-6,  # μmol/L
        hbr=100e-6,
    )

    epochs = mne.Epochs(
        raw_hb,
        events,
        task_ids,
        tmin=-5,
        tmax=15,
        baseline=(None, 0),
        reject=reject_criteria,
        preload=True,
    )
    # Drop epochs with excessive amplitude
    epochs.drop_bad()

    mo.md(f"**Epochs after artifact rejection:** {len(epochs)}")
    mo.md(f"**Epochs created:** {len(epochs)} / {len(events)}")
    epochs.plot(n_epochs=5, block=True)
    plt.show()
    return (epochs,)


@app.cell
def _(epochs, plt):
    epochs["task_10"].plot_image(
        combine="mean",
        vmin=-1.5,
        vmax=1.5,
        ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])),
    )
    plt.show()
    return


@app.cell
def _(epochs, plt):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 6), layout="constrained")
    clims = dict(hbo=[-3, 2], hbr=[-2, 2])
    epochs["task_10"].average().plot_image(axes=axes[:, 0], clim=clims)
    for column, _condition in enumerate(["Control", "Tapping"]):
        for ax in axes[:, column]:
            ax.set_title(f"{_condition}: {ax.get_title()}")

    plt.show()
    return


@app.cell
def _(epochs, mne, plt):
    evoked_dict = {
        "Tapping/HbO": epochs["task_8"].average(picks="hbo"),
        "Tapping/HbR": epochs["task_8"].average(picks="hbr"),
    }

    # Rename channels until the encoding of frequency in ch_name is fixed
    for condition in evoked_dict:
        evoked_dict[condition].rename_channels(lambda x: x[:-4])

    color_dict = dict(HbO="#AA3377", HbR="b")

    mne.viz.plot_compare_evokeds(
        evoked_dict, combine="mean", ci=0.95, colors=color_dict
    )
    plt.show()
    return


@app.cell
def _(epochs, np, plt):
    times = np.arange(-3.5, 13.2, 3.0)
    topomap_args = dict(extrapolate="local")
    epochs["task_8"].average(picks="hbo").plot_joint(
        times=times, topomap_args=topomap_args
    )
    plt.show()
    return


if __name__ == "__main__":
    app.run()
