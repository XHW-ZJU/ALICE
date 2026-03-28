import pandas as pd

df = pd.read_csv('../output/EA/EA_Top.csv').drop_duplicates(subset='AA_sequence')
df['multi_tar'] = df['Target_LY6C1_Enri']*0.5 + df['Target_LY6A_Enri']*0.5
df = df.sort_values(by='Target_LY6C1_Enri', ascending=False).reset_index(drop=True)

df = df[df['multi_tar'] >= -1]

df.to_csv('../output/EA/multi_tar_TOP.csv', index=False)
