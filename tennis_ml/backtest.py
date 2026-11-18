"""Historical odds matching and Kelly criterion profitability backtester."""

import numpy as np
import pandas as pd

from .betting import closing_line_value, kelly_fraction, remove_vig
from .features import FEATURES


def attach_odds(
    df_te,
    odds_paths,
    bet_book=("B365W", "B365L"),
    close_book=("PSW", "PSL"),
):
    """Join historical Excel/CSV odds files onto positive-target test match rows."""
    if isinstance(odds_paths, str):
        odds_paths = [odds_paths]
    frames = [
        pd.read_excel(p) if str(p).endswith(("xls", "xlsx")) else pd.read_csv(p)
        for p in odds_paths
    ]
    odds = pd.concat(frames, ignore_index=True)

    def key(name):
        parts = str(name).strip().split()
        if len(parts) < 2:
            return str(name).strip().lower()
        return f"{' '.join(parts[1:])} {parts[0][0]}.".lower()

    t = df_te[df_te["target"] == 1].copy()
    t["wk"] = t["meta_p1"].apply(key)
    t["lk"] = t["meta_p2"].apply(key)
    t["year"] = pd.to_datetime(t["meta_date"]).dt.year

    odds["wk"] = odds["Winner"].astype(str).str.strip().str.lower()
    odds["lk"] = odds["Loser"].astype(str).str.strip().str.lower()
    odds["year"] = pd.to_datetime(odds["Date"], errors="coerce").dt.year

    cols = ["wk", "lk", "year", bet_book[0], bet_book[1], close_book[0], close_book[1]]
    merged = t.merge(
        odds[cols].drop_duplicates(["wk", "lk", "year"]),
        on=["wk", "lk", "year"],
        how="left",
    )

    merged["odds_p1"] = merged[bet_book[0]]
    merged["odds_p2"] = merged[bet_book[1]]
    merged["closing_odds_p1"] = merged[close_book[0]]
    merged["closing_odds_p2"] = merged[close_book[1]]
    merged["target"] = 1

    for col in ["odds_p1", "odds_p2", "closing_odds_p1", "closing_odds_p2"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged.loc[merged[col] <= 1.01, col] = np.nan

    rate = merged["odds_p1"].notna().mean()
    print(
        f"\nCote atasate: {int(merged['odds_p1'].notna().sum())}/{len(merged)} "
        f"meciuri ({rate * 100:.1f}%)"
    )
    keep = ["odds_p1", "odds_p2", "closing_odds_p1", "closing_odds_p2"]
    out = merged.dropna(subset=keep)
    return out[FEATURES + keep + ["target"]].reset_index(drop=True)


def backtest(
    test_df,
    model,
    kelly_frac=0.25,
    min_edge=0.03,
    bankroll=1000.0,
    max_bet_pct=0.05,
):
    """Simulate fractional Kelly betting on test matches with attached market odds."""
    history, clv_list = [], []
    n_bets, n_wins = 0, 0
    start_bankroll = bankroll

    for _, row in test_df.iterrows():
        x_row = row[FEATURES].to_frame().T.astype(float)
        model_p1 = model.predict_proba(x_row)[0, 1]
        model_p2 = 1.0 - model_p1
        fair_p1, fair_p2, _ = remove_vig(row["odds_p1"], row["odds_p2"])

        for side_p, fair_p, odds, close_odds, win_target in [
            (model_p1, fair_p1, row["odds_p1"], row["closing_odds_p1"], 1),
            (model_p2, fair_p2, row["odds_p2"], row["closing_odds_p2"], 0),
        ]:
            edge = side_p - fair_p
            if edge < min_edge:
                continue
            f = kelly_fraction(side_p, odds) * kelly_frac
            stake = min(bankroll * f, bankroll * max_bet_pct)
            if stake <= 0:
                continue
            won = row["target"] == win_target
            pnl = stake * (odds - 1.0) if won else -stake
            bankroll += pnl
            n_bets += 1
            n_wins += int(won)
            clv_list.append(closing_line_value(odds, close_odds))
            history.append(
                {
                    "bankroll": bankroll,
                    "pnl": pnl,
                    "edge": edge,
                    "stake": stake,
                    "won": won,
                }
            )

    if not history:
        print("Niciun pariu cu edge suficient.")
        return pd.DataFrame()

    hist = pd.DataFrame(history)
    total_staked = hist["stake"].sum()
    roi = hist["pnl"].sum() / total_staked if total_staked else 0.0
    peak = hist["bankroll"].cummax()
    max_dd = ((hist["bankroll"] - peak) / peak).min()

    print("\n--- Backtest ---")
    print(f"Pariuri plasate:  {n_bets}")
    print(f"Win rate pariuri: {n_wins / n_bets * 100:.1f}%")
    print(f"ROI / Yield:      {roi * 100:+.2f}%")
    print(f"Bankroll final:   {bankroll:.0f} (start {start_bankroll:.0f})")
    print(f"Max drawdown:     {max_dd * 100:.1f}%")
    print(f"CLV mediu:        {np.mean(clv_list) * 100:+.2f}%  <- cel mai important")
    return hist
