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

    return ast, np, pl


@app.cell
def _(pl):
    DATA_FILE = "./data/processed/combined/combined_100hz.parquet"

    df = pl.read_parquet(DATA_FILE)
    df
    return


@app.cell
def _(ast, np):
    def unflattern_HT(flat):
        flat = ast.literal_eval(flat)
        T = np.array(flat).reshape(4, 4)
        T = T.T
        return T

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
