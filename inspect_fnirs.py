import polars as pl
df = pl.read_parquet('data/processed/combined/data_packet/fnirs_full.parquet')
print('Shape:', df.shape)
print('\nSchema:')
for c in df.columns:
    print(f'  {c}: {df[c].dtype}')
print('\nUnique runs:', df['run_id'].unique().sort().to_list())
print('Unique participants:', df['participant'].unique().sort().to_list())
print('Has is_robot:', 'is_robot' in df.columns)
print('Has task_instance:', 'task_instance' in df.columns)
print('Task instances:', df['task_instance'].n_unique())
print('\nRows per run:')
for rid in df['run_id'].unique().sort().to_list():
    r = df.filter(pl.col('run_id') == rid)
    lab = r['is_robot'][0]
    ti = r['task_instance'].n_unique()
    print(f'  {rid}: label={lab}, {ti} instances, {r.shape[0]} rows')
print('\nFirst 3 rows:')
print(df.head(3))
