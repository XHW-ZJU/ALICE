import pandas as pd

df = pd.read_csv('../output/EA/FE_score_Top500.csv')
df = df.sort_values(by='Prod_Fit', ascending=False).head(100)
df.to_csv('../output/EA/Prod_Fit_Top100.csv', index=False)
