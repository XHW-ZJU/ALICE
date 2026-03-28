"""
Date: 2023-06-17
Project: AGI-AAV Sequence Processing and Filtering
"""
import pandas as pd
df = pd.read_csv('../input/RankFiltPro/ROBERTA_Total_seq.csv').drop_duplicates(subset='AA_sequence')
df['filter_score'] = 0.3*df['Prod_Fit']+0.35*df['Target_LY6A_Enri']+0.35*df['Target_LY6C1_Enri']
df = df.sort_values(by='filter_score', ascending=False).head(1000)
df.to_csv('../output/RankFiltPro/ROBERTA_Total_seq_filter1000.csv', index=False)





