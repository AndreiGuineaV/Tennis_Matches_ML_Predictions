"""
============================================================
 Sistem de predictie meciuri ATP + evaluare value betting
 Pipeline complet, intr-un singur fisier:
   1. Feature engineering leak-free (Elo, surface Elo, serviciu EWMA, H2H, fatigue)
   2. Antrenare XGBoost cu split temporal + early stopping
   3. Calibrare izotonica a probabilitatilor
   4. Evaluare (acuratete, log loss, Brier, curba de calibrare)
   5. Backtest profitabilitate (vig removal, edge, Kelly, CLV, drawdown)
   6. Predictie live pentru un meci viitor
 Date: format Jeff Sackmann (tennis_atp), CSV 'merged_matches.csv'
============================================================
"""

from datetime import date

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
try:
    from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
except ImportError:
    FrozenEstimator = None  # sklearn <= 1.5 foloseste cv='prefit'

# ------------------------------------------------------------
# CONSTANTE
# ------------------------------------------------------------
DEFAULT_ELO = 1500
EWMA_ALPHA = 0.2          # ponderea formei recente in statisticile de serviciu
DEFAULT_SRV = {'ace_rate': 0.07, 'df_rate': 0.04,
               'first_in': 0.62, 'first_won': 0.72, 'second_won': 0.50}

FEATURES = [
    'elo_diff', 'surface_elo_diff', 'rank_diff', 'fatigue_diff', 'age_diff',
    'ace_rate_diff', 'first_won_diff', 'second_won_diff', 'first_in_diff',
    'serve_dominance', 'h2h_winrate', 'h2h_confidence', 'best_of',
]

# ------------------------------------------------------------
# ELO
# ------------------------------------------------------------
def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, score_a, k):
    exp_a = expected_score(rating_a, rating_b)
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1 - score_a) - (1 - exp_a))
    return new_a, new_b

def dynamic_k(n_matches):
    # K mare cand jucatorul are putine meciuri (rating instabil), mic dupa stabilizare
    return 250 / ((n_matches + 5) ** 0.4)

# ------------------------------------------------------------
# BETTING HELPERS
# ------------------------------------------------------------
def remove_vig(odds_p1, odds_p2):
    raw_p1, raw_p2 = 1 / odds_p1, 1 / odds_p2
    overround = raw_p1 + raw_p2
    return raw_p1 / overround, raw_p2 / overround, overround

def closing_line_value(odds_taken, closing_odds):
    fair_taken = 1 / odds_taken
    fair_close = 1 / closing_odds
    return (fair_close - fair_taken) / fair_taken

def kelly_fraction(prob, odds):
    b = odds - 1
    f = (prob * b - (1 - prob)) / b
    return max(0.0, f)

# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------
def build_features(a, b, best_of):
    """a, b: dict cu starea fiecarui jucator INAINTE de meci."""
    return {
        'elo_diff':         a['elo'] - b['elo'],
        'surface_elo_diff': a['selo'] - b['selo'],
        'rank_diff':        np.log(b['rank']) - np.log(a['rank']),
        'fatigue_diff':     a['fatigue'] - b['fatigue'],
        'age_diff':         a['age'] - b['age'],
        'ace_rate_diff':    a['srv']['ace_rate'] - b['srv']['ace_rate'],
        'first_won_diff':   a['srv']['first_won'] - b['srv']['first_won'],
        'second_won_diff':  a['srv']['second_won'] - b['srv']['second_won'],
        'first_in_diff':    a['srv']['first_in'] - b['srv']['first_in'],
        'serve_dominance':  a['srv']['first_won'] - b['srv']['second_won'],
        'h2h_winrate':      (a['h2h_wins'] + 2) / (a['h2h_total'] + 4),
        'h2h_confidence':   a['h2h_total'],
        'best_of':          best_of,
    }


class StateTracker:
    """Tine starea evolutiva a tuturor jucatorilor (Elo, serviciu, fatigue, H2H)."""
    def __init__(self):
        self.overall_elo = {}
        self.surface_elo = {'Hard': {}, 'Clay': {}, 'Grass': {}, 'Carpet': {}}
        self.match_count = {}
        self.match_dates = {}
        self.serve = {}
        self.h2h = {}

    def get_state(self, player, surface, date, rank, age):
        elo = self.overall_elo.get(player, DEFAULT_ELO)
        selo = self.surface_elo[surface].get(player, elo)  # cold start = elo general
        fatigue = sum(1 for d in self.match_dates.get(player, [])
                      if 0 <= (date - d).days <= 14)
        srv = self.serve.get(player, dict(DEFAULT_SRV))
        return {'elo': elo, 'selo': selo, 'fatigue': fatigue,
                'rank': rank, 'age': age, 'srv': srv}

    def h2h_record(self, a, b):
        key = tuple(sorted([a, b]))
        rec = self.h2h.get(key, {})
        a_wins = rec.get(a, 0)
        b_wins = rec.get(b, 0)
        return a_wins, b_wins, a_wins + b_wins

    def update_after_match(self, winner, loser, surface, date, row):
        # Elo cu K dinamic
        w_elo = self.overall_elo.get(winner, DEFAULT_ELO)
        l_elo = self.overall_elo.get(loser, DEFAULT_ELO)
        w_selo = self.surface_elo[surface].get(winner, w_elo)
        l_selo = self.surface_elo[surface].get(loser, l_elo)
        k = (dynamic_k(self.match_count.get(winner, 0)) +
             dynamic_k(self.match_count.get(loser, 0))) / 2
        self.overall_elo[winner], self.overall_elo[loser] = update_elo(w_elo, l_elo, 1, k)
        self.surface_elo[surface][winner], self.surface_elo[surface][loser] = \
            update_elo(w_selo, l_selo, 1, k)
        self.match_count[winner] = self.match_count.get(winner, 0) + 1
        self.match_count[loser] = self.match_count.get(loser, 0) + 1

        # Serviciu EWMA (leak-free: actualizam DUPA ce am citit features)
        for p, pref in [(winner, 'w_'), (loser, 'l_')]:
            svpt = row.get(pref + 'svpt', np.nan)
            if not pd.isna(svpt) and svpt > 0:
                first_in = row.get(pref + '1stIn', 0)
                new = {
                    'ace_rate':   row.get(pref + 'ace', 0) / svpt,
                    'df_rate':    row.get(pref + 'df', 0) / svpt,
                    'first_in':   first_in / svpt,
                    'first_won':  row.get(pref + '1stWon', 0) / max(first_in, 1),
                    'second_won': row.get(pref + '2ndWon', 0) / max(svpt - first_in, 1),
                }
                old = self.serve.get(p)
                self.serve[p] = new if old is None else {
                    kk: EWMA_ALPHA * new[kk] + (1 - EWMA_ALPHA) * old[kk] for kk in new
                }

        # Fatigue + pruning la 14 zile
        for p in (winner, loser):
            self.match_dates.setdefault(p, []).append(date)
            self.match_dates[p] = [d for d in self.match_dates[p]
                                   if (date - d).days <= 14]

        # H2H
        key = tuple(sorted([winner, loser]))
        rec = self.h2h.get(key, {})
        rec[winner] = rec.get(winner, 0) + 1
        rec.setdefault(loser, 0)
        self.h2h[key] = rec


def prepare_ml_dataset(matches_df):
    print("Construim dataset-ul (Elo, serviciu, H2H, fatigue) in ordine cronologica...")
    matches_df = matches_df.copy()
    matches_df['tourney_date'] = pd.to_datetime(matches_df['tourney_date'])
    sort_cols = ['tourney_date'] + (['match_num'] if 'match_num' in matches_df.columns else [])
    matches_df = matches_df.sort_values(sort_cols).reset_index(drop=True)

    st = StateTracker()
    rows = []

    for _, r in matches_df.iterrows():
        if pd.isna(r['winner_name']) or pd.isna(r['loser_name']) or pd.isna(r['surface']):
            continue
        w, l, surf, date = r['winner_name'], r['loser_name'], r['surface'], r['tourney_date']
        if surf not in st.surface_elo:
            continue

        w_rank = 500 if pd.isna(r.get('winner_rank', np.nan)) else r.get('winner_rank')
        l_rank = 500 if pd.isna(r.get('loser_rank', np.nan)) else r.get('loser_rank')
        w_age = r.get('winner_age', 26) if not pd.isna(r.get('winner_age', np.nan)) else 26
        l_age = r.get('loser_age', 26) if not pd.isna(r.get('loser_age', np.nan)) else 26
        best_of = r.get('best_of', 3)

        w_state = st.get_state(w, surf, date, w_rank, w_age)
        l_state = st.get_state(l, surf, date, l_rank, l_age)
        w_h2h, l_h2h, total_h2h = st.h2h_record(w, l)
        w_state['h2h_wins'], w_state['h2h_total'] = w_h2h, total_h2h
        l_state['h2h_wins'], l_state['h2h_total'] = l_h2h, total_h2h

        # OGLINDIRE: fiecare meci de doua ori -> simetrie + dublu de date
        row_w = build_features(w_state, l_state, best_of); row_w['target'] = 1
        row_l = build_features(l_state, w_state, best_of); row_l['target'] = 0
        # metadate (NU sunt features) - necesare pentru a atasa cotele
        for rr, p1n, p2n in [(row_w, w, l), (row_l, l, w)]:
            rr['meta_date'] = date; rr['meta_p1'] = p1n; rr['meta_p2'] = p2n
        rows.append(row_w)
        rows.append(row_l)

        st.update_after_match(w, l, surf, date, r)

    df = pd.DataFrame(rows)
    print(f"Dataset gata: {len(df)} randuri ({len(df)//2} meciuri x 2 orientari).")
    return df, st


# ------------------------------------------------------------
# ANTRENARE + CALIBRARE + EVALUARE
# ------------------------------------------------------------
def train_model(df_ml):
    X, Y = df_ml[FEATURES], df_ml['target']
    # split PE MECI (randurile oglindite sunt consecutive) ca sa evitam leakage
    n_matches = len(df_ml) // 2
    i1, i2 = int(n_matches * 0.70) * 2, int(n_matches * 0.85) * 2
    X_tr, X_val, X_te = X.iloc[:i1], X.iloc[i1:i2], X.iloc[i2:]
    y_tr, y_val, y_te = Y.iloc[:i1], Y.iloc[i1:i2], Y.iloc[i2:]

    print("\nAntrenam XGBoost (early stopping pe log loss)...")
    model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        n_estimators=3000, learning_rate=0.02, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=6,
        reg_lambda=1.5, early_stopping_rounds=80, random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    print("Calibram probabilitatile (izotonic)...")
    if FrozenEstimator is not None:
        # sklearn >= 1.6: 'cv=prefit' a fost eliminat -> folosim FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='isotonic')
    else:
        # sklearn <= 1.5: varianta veche
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated.fit(X_val, y_val)

    proba = calibrated.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print("\n--- Evaluare pe setul de test ---")
    print(f"Acuratete: {accuracy_score(y_te, pred) * 100:.2f}%")
    print(f"Log loss:  {log_loss(y_te, proba):.4f}")
    print(f"Brier:     {brier_score_loss(y_te, proba):.4f}")
    frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=10)
    print("Calibrare (prezis -> real):")
    for mp, fp in zip(mean_pred, frac_pos):
        print(f"  {mp*100:4.0f}% -> {fp*100:4.0f}%")

    df_te = df_ml.iloc[i2:].reset_index(drop=True)
    return calibrated, df_te

def attach_odds(df_te, odds_paths,
                bet_book=('B365W', 'B365L'),   # cota la care "pariezi"
                close_book=('PSW', 'PSL')):     # Pinnacle ~ linia de inchidere (CLV)
    if isinstance(odds_paths, str):
        odds_paths = [odds_paths]
    frames = [pd.read_excel(p) if str(p).endswith(('xls', 'xlsx')) else pd.read_csv(p)
              for p in odds_paths]
    odds = pd.concat(frames, ignore_index=True)

    def key(name):
        parts = str(name).strip().split()
        if len(parts) < 2:
            return str(name).strip().lower()
        return f"{' '.join(parts[1:])} {parts[0][0]}.".lower()  # 'Jannik Sinner' -> 'sinner j.'

    t = df_te[df_te['target'] == 1].copy()
    t['wk'] = t['meta_p1'].apply(key)
    t['lk'] = t['meta_p2'].apply(key)
    t['year'] = pd.to_datetime(t['meta_date']).dt.year

    odds['wk'] = odds['Winner'].astype(str).str.strip().str.lower()
    odds['lk'] = odds['Loser'].astype(str).str.strip().str.lower()
    odds['year'] = pd.to_datetime(odds['Date'], errors='coerce').dt.year

    cols = ['wk', 'lk', 'year', bet_book[0], bet_book[1], close_book[0], close_book[1]]
    merged = t.merge(odds[cols].drop_duplicates(['wk', 'lk', 'year']),
                     on=['wk', 'lk', 'year'], how='left')

    merged['odds_p1'] = merged[bet_book[0]]
    merged['odds_p2'] = merged[bet_book[1]]
    merged['closing_odds_p1'] = merged[close_book[0]]
    merged['closing_odds_p2'] = merged[close_book[1]]
    merged['target'] = 1

    # cotele citite din Excel pot veni ca text -> le facem numerice
    for c in ['odds_p1', 'odds_p2', 'closing_odds_p1', 'closing_odds_p2']:  
        merged[c] = pd.to_numeric(merged[c], errors='coerce')

    # cotele 0 / sub 1.01 sunt date lipsa, nu cote reale -> le scoatem
    for c in ['odds_p1', 'odds_p2', 'closing_odds_p1', 'closing_odds_p2']:
        merged.loc[merged[c] <= 1.01, c] = np.nan
        
    rate = merged['odds_p1'].notna().mean()
    print(f"\nCote atasate: {int(merged['odds_p1'].notna().sum())}/{len(merged)} "
          f"meciuri ({rate*100:.1f}%)")
    keep = ['odds_p1', 'odds_p2', 'closing_odds_p1', 'closing_odds_p2']
    out = merged.dropna(subset=keep)
    return out[FEATURES + keep + ['target']].reset_index(drop=True)

# ------------------------------------------------------------
# BACKTEST PROFITABILITATE
# ------------------------------------------------------------
def backtest(test_df, model, kelly_frac=0.25, min_edge=0.03,
             bankroll=1000.0, max_bet_pct=0.05):
    """
    test_df: features (FEATURES) + 'odds_p1','odds_p2',
             'closing_odds_p1','closing_odds_p2','target' (1 daca p1 castiga).
    """
    history, clv_list = [], []
    n_bets, n_wins = 0, 0

    for _, row in test_df.iterrows():
        X_row = row[FEATURES].to_frame().T.astype(float)
        model_p1 = model.predict_proba(X_row)[0, 1]
        model_p2 = 1 - model_p1
        fair_p1, fair_p2, _ = remove_vig(row['odds_p1'], row['odds_p2'])

        for side_p, fair_p, odds, close_odds, win_target in [
            (model_p1, fair_p1, row['odds_p1'], row['closing_odds_p1'], 1),
            (model_p2, fair_p2, row['odds_p2'], row['closing_odds_p2'], 0),
        ]:
            edge = side_p - fair_p
            if edge < min_edge:
                continue
            f = kelly_fraction(side_p, odds) * kelly_frac
            stake = min(bankroll * f, bankroll * max_bet_pct)
            if stake <= 0:
                continue
            won = (row['target'] == win_target)
            pnl = stake * (odds - 1) if won else -stake
            bankroll += pnl
            n_bets += 1
            n_wins += int(won)
            clv_list.append(closing_line_value(odds, close_odds))
            history.append({'bankroll': bankroll, 'pnl': pnl,
                            'edge': edge, 'stake': stake, 'won': won})

    if not history:
        print("Niciun pariu cu edge suficient.")
        return pd.DataFrame()

    hist = pd.DataFrame(history)
    total_staked = hist['stake'].sum()
    roi = hist['pnl'].sum() / total_staked if total_staked else 0
    peak = hist['bankroll'].cummax()
    max_dd = ((hist['bankroll'] - peak) / peak).min()

    print("\n--- Backtest ---")
    print(f"Pariuri plasate:  {n_bets}")
    print(f"Win rate pariuri: {n_wins / n_bets * 100:.1f}%")
    print(f"ROI / Yield:      {roi * 100:+.2f}%")
    print(f"Bankroll final:   {bankroll:.0f} (start 1000)")
    print(f"Max drawdown:     {max_dd * 100:.1f}%")
    print(f"CLV mediu:        {np.mean(clv_list) * 100:+.2f}%  <- cel mai important")
    return hist


# ------------------------------------------------------------
# PREDICTIE LIVE pentru un meci viitor
# ------------------------------------------------------------
def predict_match(model, st, player_1, player_2, surface, best_of=3,
                  rank_1=500, rank_2=500, age_1=26, age_2=26,
                  odds_p1=None, odds_p2=None):
    today = pd.Timestamp.today()
    a = st.get_state(player_1, surface, today, rank_1, age_1)
    b = st.get_state(player_2, surface, today, rank_2, age_2)
    a_wins, b_wins, total = st.h2h_record(player_1, player_2)
    a['h2h_wins'], a['h2h_total'] = a_wins, total
    feats = pd.DataFrame([build_features(a, b, best_of)])[FEATURES]
    p1 = model.predict_proba(feats)[0, 1]

    print(f"\n--- {surface}: {player_1} vs {player_2} ---")
    print(f"Model: {player_1} {p1*100:.1f}% | {player_2} {(1-p1)*100:.1f}%")
    print(f"H2H: {player_1} {a_wins}-{b_wins} {player_2}")

    if odds_p1 and odds_p2:
        fair_p1, fair_p2, _ = remove_vig(odds_p1, odds_p2)
        for name, my_p, fair_p, odds in [
            (player_1, p1, fair_p1, odds_p1),
            (player_2, 1 - p1, fair_p2, odds_p2),
        ]:
            edge = my_p - fair_p
            tag = f"VALUE (+{edge*100:.1f}% edge)" if edge > 0.03 else "fara value"
            print(f"  {name}: model {my_p*100:.1f}% vs piata {fair_p*100:.1f}% (cota {odds}) -> {tag}")
    return p1


# ------------------------------------------------------------
# RULARE
# ------------------------------------------------------------
if __name__ == '__main__':
    matches_df = pd.read_csv('merged_matches.csv', low_memory=False)
    df_ml, state = prepare_ml_dataset(matches_df)
    model, df_te = train_model(df_ml)

    # --- Backtest cu cote reale ---
    import glob
    test_df = attach_odds(df_te, sorted(glob.glob("odds/tennis_odds_*.xlsx")))
    if not test_df.empty:
        backtest(test_df, model)

    # --- Predictie live ---
    predict_match(model, state,
                  player_1="Zizou Bergs", player_2="Toby Samuel",
                  surface="Grass", best_of=3,
                  rank_1=48, rank_2=142,
                  odds_p1=1.37, odds_p2=3.10)