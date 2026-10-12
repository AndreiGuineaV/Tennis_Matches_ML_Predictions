import pandas as pd

# 1. Incarca ambele fisiere
hist = pd.read_csv("merged_matches.csv")       # 1968 -> apr 2026
new  = pd.read_csv("2026.csv")     # tot sezonul 2026 (de la Sackmann)

# 2. Arunca 'indoor' (exista doar in fisierul 2026, modelul nu o foloseste)
new = new.drop(columns=['indoor'], errors='ignore')

# 3. Lipeste-le DUPA NUMELE coloanei (pandas aliniaza singur, ordinea nu conteaza)
df = pd.concat([hist, new], ignore_index=True)

# 4. Uniformizeaza data (un fisier are "2026-04-22", altul poate avea 20260422)
df['tourney_date'] = pd.to_datetime(
    df['tourney_date'].astype(str).str.replace('-', '', regex=False),
    format='%Y%m%d', errors='coerce'
)

# 5. ELIMINA DUPLICATELE - meciurile ian-apr 2026 sunt in AMBELE fisiere!
df = df.drop_duplicates(subset=['tourney_id', 'match_num'], keep='last')

# 6. SORTEAZA CRONOLOGIC (esential - altfel modelul "vede viitorul" = data leakage)
df = df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)

# 7. Salveaza peste fisierul vechi
df.to_csv("merged_matches.csv", index=False)
print(f"Total meciuri: {len(df)} | ultimul: {df['tourney_date'].max().date()}")