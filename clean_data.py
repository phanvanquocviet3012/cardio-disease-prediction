import pandas as pd
df = pd.read_csv('cardio_train.csv', sep=';') 

df['age_years'] = (df['age'] / 365.25).round().astype(int)

df_clean = df[
    (df['age_years'] >= 10) & (df['age_years'] <= 100) &
    (df['ap_lo'] >= 40) & (df['ap_lo'] <= 160) &
    (df['ap_hi'] >= 60) & (df['ap_hi'] <= 240) &
    (df['ap_hi'] > df['ap_lo']) & 
    (df['height'] >= 100) & (df['height'] <= 250) & 
    (df['weight'] >= 30) & (df['weight'] <= 200)
].copy()

df_clean.to_csv('clean_data.csv', index=False, encoding='utf-8')