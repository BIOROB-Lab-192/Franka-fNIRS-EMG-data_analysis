import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import scipy.signal as sp
    import polars as pl
    import marimo as mo
    import numpy as np

    return mo, np, pl, sp


@app.cell
def _(pl):
    emg_data = "./data/processed/combined/data_packet/emg_full.parquet"
    emg_df = pl.read_parquet(emg_data)
    return (emg_df,)


@app.cell(hide_code=True)
def _(emg_df, mo):
    # Reuses: mo, emg_df, pl

    run_select = mo.ui.dropdown(
        options=sorted(emg_df["run_id"].unique().to_list()),
        label="Run",
    )



    task_select = mo.ui.dropdown(
        options=sorted(emg_df["task_instance"].unique().to_list()),
        label="Task instance",
    )

    mo.vstack([
        run_select,
        task_select,
    ])
    return run_select, task_select


@app.function(hide_code=True)
def plot_emg(df):
    import matplotlib.pyplot as plt

    emg_cols = [
        c for c in df.columns
        if "EMG" in c and c.endswith("(mV)")
    ]

    fig, axes = plt.subplots(
        len(emg_cols),
        1,
        figsize=(14, 2.5 * len(emg_cols)),
        sharex=True,
    )

    if len(emg_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, emg_cols):
        sensor = col.replace(" (mV)", "")

        ax.plot(df["time_sec"], df[col], linewidth=0.6)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel(sensor, fontsize=8)

    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    # plt.show()

    return fig


@app.cell(hide_code=True)
def _(emg_df, mo, np, pl, run_select, task_select):
    # Reuses: emg_df, pl, plot_emg

    filtered_emg = (
        emg_df
        .filter(
            (pl.col("run_id") == run_select.value)
            & (pl.col("task_instance") == task_select.value)
        )
        .sort("time_sec")
    )

    if filtered_emg.height == 0:
        output = mo.md("No EMG data matches this selection.")
    else:
        _fig = plot_emg(filtered_emg)

    output = _fig
    time = filtered_emg["time_sec"].to_numpy() 
    fs = 1 / np.median(np.diff(time))
    output
    return filtered_emg, fs


@app.cell
def _(filtered_emg, fs, pl, sp):
    b, a = sp.butter(
        2,
        20,
        btype="highpass",
        fs=fs,
    )
    emg_cols = [
        c for c in filtered_emg.columns
        if "EMG" in c and c.endswith("(mV)")
    ]

    new_filtered_df = filtered_emg.with_columns([
        pl.Series(col, sp.filtfilt(b, a, filtered_emg[col].fill_nan(None).interpolate().fill_null(strategy="forward").fill_null(strategy="backward").to_numpy()))
        for col in emg_cols
    ])
    plot_emg(new_filtered_df)
    return (emg_cols,)


@app.cell
def _(filtered_emg, fs, pl, sp):
    def _():
        b, a = sp.butter(
            2,
            [20, 450],
            btype="bandpass",
            fs=fs,
        )
        emg_cols = [
            c for c in filtered_emg.columns
            if "EMG" in c and c.endswith("(mV)")
        ]

        new_filtered_df = filtered_emg.with_columns([
            pl.Series(col, sp.filtfilt(b, a, filtered_emg[col].fill_nan(None).interpolate().fill_null(strategy="forward").fill_null(strategy="backward").to_numpy()))
            for col in emg_cols
        ])
        return plot_emg(new_filtered_df)


    _()
    return


@app.function(hide_code=True)
def reduce_emg_noise_with_baseline(x, time, fs, baseline_start=-5, baseline_end=0, noise_factor=1.5, gain_floor=0.15):
    import numpy as np
    import scipy.signal as sp

    x = np.asarray(x)
    time = np.asarray(time)

    baseline_mask = (time >= baseline_start) & (time <= baseline_end)

    if baseline_mask.sum() < int(0.5 * fs):
        return x  # not enough baseline; return unchanged

    nperseg = int(0.25 * fs)   # 250 ms window
    noverlap = int(0.20 * fs)  # 200 ms overlap

    f, t_stft, Z = sp.stft(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    _, _, Z_noise = sp.stft(
        x[baseline_mask],
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    noise_mag = np.median(np.abs(Z_noise), axis=1, keepdims=True)

    mag = np.abs(Z)
    phase = np.exp(1j * np.angle(Z))

    cleaned_mag = mag - noise_factor * noise_mag
    cleaned_mag = np.maximum(cleaned_mag, gain_floor * mag)

    Z_clean = cleaned_mag * phase

    _, y = sp.istft(
        Z_clean,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    return y[:len(x)]


@app.cell
def _(filtered_emg, np, pl):
    def _():
        time = filtered_emg["time_sec"].to_numpy()
        fs = 1 / np.median(np.diff(time))

        emg_cols = [
            c for c in filtered_emg.columns
            if "EMG" in c and c.endswith("(mV)")
        ]

        new_filtered_df = filtered_emg.with_columns([
            pl.Series(
                col,
                reduce_emg_noise_with_baseline(
                    filtered_emg[col]
                    .fill_nan(None)
                    .interpolate()
                    .fill_null(strategy="forward")
                    .fill_null(strategy="backward")
                    .to_numpy(),
                    time,
                    fs,
                    noise_factor=2,
                    gain_floor=0.15
                )
            )
            for col in emg_cols
        ])
        return plot_emg(new_filtered_df)


    _()
    return


@app.cell
def _(emg_df, mo, np, pl, run_select, sp, task_select):
    rms_df = None
    def _():
        global rms_df
        # Reuses: emg_df, pl, plot_emg

        filtered_emg = (
            emg_df
            .filter(
                (pl.col("run_id") == run_select.value)
                & (pl.col("task_instance") == task_select.value)
            )
            .sort("time_sec")
        )

        if filtered_emg.height == 0:
            output = mo.md("No EMG data matches this selection.")
        else:
            emg_cols = [
                c for c in filtered_emg.columns
                if "EMG" in c and c.endswith("(mV)")
            ]

            time = filtered_emg["time_sec"].to_numpy()
            fs = 1 / np.median(np.diff(time))

            # Band-pass first
            sos = sp.butter(
                2,
                [20, 450],
                btype="bandpass",
                fs=fs,
                output="sos",
            )

            bandpass_df = filtered_emg.with_columns([
                pl.Series(
                    col,
                    sp.sosfiltfilt(
                        sos,
                        filtered_emg[col]
                        .fill_nan(None)
                        .interpolate()
                        .fill_null(strategy="forward")
                        .fill_null(strategy="backward")
                        .to_numpy()
                    )
                )
                for col in emg_cols
            ])

            # RMS second
            rms_window_sec = 0.10  # 100 ms
            rms_window_samples = int(rms_window_sec * fs)

            rms_df = bandpass_df.with_columns([
                (
                    pl.col(col)
                    .pow(2)
                    .rolling_mean(
                        window_size=rms_window_samples,
                        center=True,
                    )
                    .sqrt()
                    .alias(col)
                )
                for col in emg_cols
            ])

            output = plot_emg(rms_df)
        return output


    _()
    return (rms_df,)


@app.cell
def _(emg_cols, pl, rms_df):
    baseline_mask = (pl.col("time_sec") >= -5) & (pl.col("time_sec") <= 0)

    baseline_values = rms_df.filter(baseline_mask).select([
        pl.col(col).mean().alias(col)
        for col in emg_cols
    ])

    rms_baselined_df = rms_df.with_columns([
        (
            pl.col(col) - baseline_values[col][0]
        ).alias(col)
        for col in emg_cols
    ])

    plot_emg(rms_baselined_df)
    return


if __name__ == "__main__":
    app.run()
