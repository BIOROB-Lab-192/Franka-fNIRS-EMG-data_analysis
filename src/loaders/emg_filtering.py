"""Takes the emg dataframe and returns a new dataframe that """

from scipy.signal import butter, sosfiltfilt
import polars as pl
import numpy as np

def filter_rms(emg_df):

    time = emg_df["time_sec"].to_numpy() 
    fs = 1 / np.median(np.diff(time))
    
    emg_cols = [
        c for c in emg_df.columns
        if "EMG" in c and c.endswith("(mV)")
    ]

    sos = butter(
                2,
                [20, 450],
                btype="bandpass",
                fs=fs,
                output="sos",
            )

    bandpass_df = emg_df.with_columns([
                pl.Series(
                    col,
                    sosfiltfilt(
                        sos,
                        emg_df[col]
                        .fill_nan(None)
                        .interpolate()
                        .fill_null(strategy="forward")
                        .fill_null(strategy="backward")
                        .to_numpy()
                    )
                )
                for col in emg_cols
            ])

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

    return rms_df