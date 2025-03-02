import pandas as pd
import numpy as np
from pathlib import Path

expert = pd.read_excel('results/Results_expert+AI.xlsx')

dfs = []
for version in Path('lightning_logs/0.0_wb_new').iterdir():
    df = pd.concat([pd.read_csv(file, index_col=0) for file in version.glob('**/easi.csv')])
    df.index = df.index.str.replace('.jpg', '')
    df = df.reindex(expert.Photo)
    df['EASI_score'] = df.sum(axis=1)
    df['Weight'] = expert.Weight
    dfs.append(df)
with pd.ExcelWriter('results/Results_AI.xlsx') as writer:
    for i, df in enumerate(dfs):
        df.to_excel(writer, sheet_name=f'Sheet{i+1}')
    