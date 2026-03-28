# Remove sequences that appeared in previous patents
import pandas as pd
df = pd.read_csv('../output/EA/multi_tar_TOP.csv')
df = df[~df['AA_sequence'].str.contains('IRQGYSS', na=False)]
df = df[~df['AA_sequence'].str.contains('LRVGYSS', na=False)]
df = df[~df['AA_sequence'].str.contains('LRAGYSS', na=False)]
df.to_csv('../output/EA/ALICE_TOP.csv', index=False)
