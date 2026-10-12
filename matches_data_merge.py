import pandas as pd
import glob
import os

def load_and_merge_data(folder_path):
    print("Începem încărcarea și îmbinarea fișierelor CSV...")
    
    # 1. Găsim toate fișierele care respectă tiparul numelui
    # Presupunem că fișierele se numesc "atp_matches_1968.csv", etc.
    file_pattern = os.path.join(folder_path, "atp_matches_[1-2]*.csv")
    csv_files = glob.glob(file_pattern)
    
    if not csv_files:
        print("Eroare: Nu am găsit niciun fișier CSV în folderul specificat!")
        return None

    # 2. Citim fiecare CSV și îl adăugăm într-o listă
    dataframes = []
    for file in csv_files:
        try:
            # Citim fișierul
            df_year = pd.read_csv(file)
            dataframes.append(df_year)
            print(f"Încărcat cu succes: {os.path.basename(file)}")
        except Exception as e:
            print(f"Eroare la citirea fișierului {file}: {e}")

    # 3. Unim toate DataFrame-urile într-unul singur
    matches_df = pd.concat(dataframes, ignore_index=True)
    
    # 4. Formatarea datei (Jeff Sackmann folosește formatul YYYYMMDD, ex: 20260508)
    # Trebuie să transformăm acest număr într-un format de tip DateTime real
    matches_df['tourney_date'] = pd.to_datetime(matches_df['tourney_date'], format='%Y%m%d', errors='coerce')
    
    # 5. ORDONAREA CRONOLOGICĂ (Crucial pentru ELO!)
    # Sortăm mai întâi după data turneului și apoi după numărul meciului din turneu
    matches_df = matches_df.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)
    
    print("\n--- Rezumat ---")
    print(f"Număr total de meciuri încărcate: {len(matches_df)}")
    print(f"Perioada acoperită: {matches_df['tourney_date'].min().date()} -> {matches_df['tourney_date'].max().date()}")
    
    return matches_df

# --- Cum folosești funcția ---
# Înlocuiește 'calea/catre/folderul/tau' cu folderul unde ai pus fisierele CSV
# Ex: folder_path = './data' 
folder_path = 'C:\\Users\\AndreiGuinea\\OneDrive\\Documente\\Python Projects\\Tennis_Matches_ML_Predictions\\atp_matches' 

matches_df = load_and_merge_data(folder_path)

matches_df.to_csv('merged_matches.csv', index=False)  