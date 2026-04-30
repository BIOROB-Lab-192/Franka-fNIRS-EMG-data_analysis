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
        """
        Load a Trigno mixed CSV into:
          - main_df: the main signal table only, with unique column names using sensor labels
          - marker_df: the marker/event table from the right side

        Assumptions based on the provided file structure:
          - The main header row is the row containing repeated 'Time Series' labels.
          - Two rows above that header is the sensor-name row.
          - The marker table begins at the columns whose headers start with:
                Type, Name, Label, Time (s)
          - The two rows below the main header are frequency and delta-t rows, not data.
        """
        filepath = Path(filepath)

        with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
            rows = [[cell.strip() for cell in row] for row in csv.reader(f)]

        # Find the main header row
        header_idx = None
        for i, row in enumerate(rows):
            if sum("Time Series" in cell for cell in row) >= 2:
                header_idx = i
                break

        if header_idx is None:
            raise ValueError("Could not find the main signal header row.")

        if header_idx < 2:
            raise ValueError("Could not find the sensor-name row above the main header.")

        sensor_row_idx = header_idx - 2
        sensor_row = rows[sensor_row_idx]
        header_row = rows[header_idx]

        # Find where the marker table starts on the right
        marker_signature = ["Type", "Name", "Label", "Time (s)"]
        marker_start_idx = None

        for j in range(len(header_row) - len(marker_signature) + 1):
            if header_row[j:j + len(marker_signature)] == marker_signature:
                marker_start_idx = j
                break

        if marker_start_idx is None:
            marker_start_idx = len(header_row)

        # Build unique main-table column names using the sensor row
        # The current sensor label carries forward until the next non-empty sensor cell.
        current_sensor = None
        raw_main_columns = []
        main_indices = []

        for j in range(marker_start_idx):
            sensor_cell = sensor_row[j] if j < len(sensor_row) else ""
            header_cell = header_row[j] if j < len(header_row) else ""

            if sensor_cell:
                current_sensor = sensor_cell

            if header_cell:
                if current_sensor:
                    col_name = f"{current_sensor} | {header_cell}"
                else:
                    col_name = header_cell
                raw_main_columns.append(col_name)
                main_indices.append(j)

        # Make names unique in case the same sensor/header pair appears more than once
        counts = {}
        main_columns = []
        for name in raw_main_columns:
            counts[name] = counts.get(name, 0) + 1
            if counts[name] == 1:
                main_columns.append(name)
            else:
                main_columns.append(f"{name}__{counts[name]}")

        # Marker table columns
        marker_indices = [j for j in range(marker_start_idx, len(header_row)) if header_row[j] != ""]
        marker_columns_raw = [header_row[j] for j in marker_indices]

        counts = {}
        marker_columns = []
        for name in marker_columns_raw:
            counts[name] = counts.get(name, 0) + 1
            if counts[name] == 1:
                marker_columns.append(name)
            else:
                marker_columns.append(f"{name}__{counts[name]}")

        # Data starts after:
        #   main header row
        #   frequency row
        #   delta-t row
        data_start_idx = header_idx + 3

        main_rows = []
        marker_rows = []

        for row in rows[data_start_idx:]:
            if not any(cell != "" for cell in row):
                continue

            main_part = [row[j] if j < len(row) else "" for j in main_indices]
            marker_part = [row[j] if j < len(row) else "" for j in marker_indices]

            if any(v != "" for v in main_part):
                main_rows.append(main_part)

            if marker_indices and any(v != "" for v in marker_part):
                marker_rows.append(marker_part)

        main_df = pl.DataFrame(main_rows, schema=main_columns, orient="row")
        marker_df = pl.DataFrame(marker_rows, schema=marker_columns, orient="row")

        # Cast the main table to numeric where possible
        main_df = main_df.with_columns(
            [
                pl.col(c).replace("", None).cast(pl.Float64, strict=False).alias(c)
                for c in main_df.columns
            ]
        )

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
