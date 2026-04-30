import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import csv
    from pathlib import Path
    import polars as pl

    return Path, csv, pl


@app.cell
def _(Path, csv, pl):
    def load_trigno_csv(filepath):
        filepath = Path(filepath)

        with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
            rows = [[cell.strip() for cell in row] for row in csv.reader(f)]

        # Find main signal header row
        main_header_idx = None
        for i, row in enumerate(rows):
            if sum("Time Series" in cell for cell in row) >= 2:
                main_header_idx = i
                break

        if main_header_idx is None:
            raise ValueError("Could not find the main signal header row.")

        if main_header_idx < 2:
            raise ValueError("Could not find the sensor-name row above the main header.")

        sensor_row = rows[main_header_idx - 2]
        main_header_row = rows[main_header_idx]

        # Find marker table header row and where it starts
        marker_header_idx = None
        marker_start_idx = None
        marker_signature = ["Type", "Name", "Label", "Time (s)"]

        for i, row in enumerate(rows):
            for j in range(len(row) - len(marker_signature) + 1):
                if row[j:j + len(marker_signature)] == marker_signature:
                    marker_header_idx = i
                    marker_start_idx = j
                    break
            if marker_header_idx is not None:
                break

        # Main columns: everything before marker block
        main_end_idx = marker_start_idx if marker_start_idx is not None else len(main_header_row)

        current_sensor = None
        raw_main_columns = []
        main_indices = []

        for j in range(main_end_idx):
            sensor_cell = sensor_row[j] if j < len(sensor_row) else ""
            header_cell = main_header_row[j] if j < len(main_header_row) else ""

            if sensor_cell:
                current_sensor = sensor_cell

            if header_cell:
                col_name = f"{current_sensor} | {header_cell}" if current_sensor else header_cell
                raw_main_columns.append(col_name)
                main_indices.append(j)

        # Make main columns unique
        counts = {}
        main_columns = []
        for name in raw_main_columns:
            counts[name] = counts.get(name, 0) + 1
            main_columns.append(name if counts[name] == 1 else f"{name}__{counts[name]}")

        # Main data starts after main header + frequency row + delta-t row
        main_data_start_idx = main_header_idx + 3

        main_rows = []
        for row in rows[main_data_start_idx:]:
            if not any(cell != "" for cell in row):
                continue

            values = [row[j] if j < len(row) else "" for j in main_indices]

            # keep only rows that actually contain main-table data
            if any(v != "" for v in values):
                main_rows.append(values)

        main_df = pl.DataFrame(main_rows, schema=main_columns, orient="row")

        # Safely cast main table to floats
        main_df = main_df.with_columns(
            [
                pl.when(pl.col(c).cast(pl.String).str.strip_chars() == "")
                .then(None)
                .otherwise(pl.col(c).cast(pl.String).str.strip_chars())
                .cast(pl.Float64, strict=False)
                .alias(c)
                for c in main_df.columns
            ]
        )

        # Marker table
        if marker_header_idx is None:
            marker_df = pl.DataFrame()
        else:
            marker_header_row = rows[marker_header_idx]
            marker_indices = [j for j in range(marker_start_idx, len(marker_header_row)) if marker_header_row[j] != ""]
            raw_marker_columns = [marker_header_row[j] for j in marker_indices]

            counts = {}
            marker_columns = []
            for name in raw_marker_columns:
                counts[name] = counts.get(name, 0) + 1
                marker_columns.append(name if counts[name] == 1 else f"{name}__{counts[name]}")

            marker_rows = []
            for row in rows[marker_header_idx + 1:]:
                if not any(cell != "" for cell in row):
                    continue

                values = [row[j] if j < len(row) else "" for j in marker_indices]

                if any(v != "" for v in values):
                    marker_rows.append(values)

            marker_df = pl.DataFrame(marker_rows, schema=marker_columns, orient="row")

        return main_df, marker_df

    return (load_trigno_csv,)


@app.cell
def _(load_trigno_csv):
    main_df, marker_df = load_trigno_csv("./data/raw/caroline_norobot1/Trial_16.csv")
    return (marker_df,)


@app.cell
def _(marker_df):
    marker_df
    return


if __name__ == "__main__":
    app.run()
