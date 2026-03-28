import pandas as pd

df = pd.read_csv('../output/EA/LY6C1_model_LY6A_model_Prod_Fit_RoBERTa_SeqGAN_RankFiltPro_FE_1000.csv')
df = df.sort_values(by='FE_score', ascending=False).head(500)
df.to_csv('../output/EA/FE_score_Top500.csv', index=False)
