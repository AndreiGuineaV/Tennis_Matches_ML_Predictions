"""Leak-free chronological feature engineering and player state tracking."""

import math
from .elo import DEFAULT_ELO, dynamic_k, update_elo

EWMA_ALPHA = 0.2
DEFAULT_SRV = {
    "ace_rate": 0.07,
    "df_rate": 0.04,
    "first_in": 0.62,
    "first_won": 0.72,
    "second_won": 0.50,
}

FEATURES = [
    "elo_diff",
    "surface_elo_diff",
    "rank_diff",
    "fatigue_diff",
    "age_diff",
    "ace_rate_diff",
    "first_won_diff",
    "second_won_diff",
    "first_in_diff",
    "serve_dominance",
    "h2h_winrate",
    "h2h_confidence",
    "best_of",
]


def _is_missing(val):
    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False


def build_features(a, b, best_of):
    """Build feature dictionary from pre-match player state snapshots a and b."""
    return {
        "elo_diff": a["elo"] - b["elo"],
        "surface_elo_diff": a["selo"] - b["selo"],
        "rank_diff": math.log(b["rank"]) - math.log(a["rank"]),
        "fatigue_diff": a["fatigue"] - b["fatigue"],
        "age_diff": a["age"] - b["age"],
        "ace_rate_diff": a["srv"]["ace_rate"] - b["srv"]["ace_rate"],
        "first_won_diff": a["srv"]["first_won"] - b["srv"]["first_won"],
        "second_won_diff": a["srv"]["second_won"] - b["srv"]["second_won"],
        "first_in_diff": a["srv"]["first_in"] - b["srv"]["first_in"],
        "serve_dominance": a["srv"]["first_won"] - a["srv"]["second_won"],
        "h2h_winrate": (a["h2h_wins"] + 2) / (a["h2h_total"] + 4),
        "h2h_confidence": a["h2h_total"],
        "best_of": best_of,
    }


class StateTracker:
    """Tracks evolving player state (Elo, surface Elo, serve EWMA, fatigue, H2H)."""

    def __init__(self):
        self.overall_elo = {}
        self.surface_elo = {"Hard": {}, "Clay": {}, "Grass": {}, "Carpet": {}}
        self.match_count = {}
        self.match_dates = {}
        self.serve = {}
        self.h2h = {}

    def get_state(self, player, surface, match_date, rank, age):
        elo = self.overall_elo.get(player, DEFAULT_ELO)
        selo = self.surface_elo[surface].get(player, elo)
        fatigue = sum(
            1 for d in self.match_dates.get(player, []) if 0 <= (match_date - d).days <= 14
        )
        srv = dict(self.serve.get(player, DEFAULT_SRV))
        return {
            "elo": elo,
            "selo": selo,
            "fatigue": fatigue,
            "rank": rank,
            "age": age,
            "srv": srv,
        }

    def h2h_record(self, a, b):
        key = tuple(sorted([a, b]))
        rec = self.h2h.get(key, {})
        a_wins = rec.get(a, 0)
        b_wins = rec.get(b, 0)
        return a_wins, b_wins, a_wins + b_wins

    def update_after_match(self, winner, loser, surface, match_date, row):
        w_elo = self.overall_elo.get(winner, DEFAULT_ELO)
        l_elo = self.overall_elo.get(loser, DEFAULT_ELO)
        w_selo = self.surface_elo[surface].get(winner, w_elo)
        l_selo = self.surface_elo[surface].get(loser, l_elo)
        k = (
            dynamic_k(self.match_count.get(winner, 0))
            + dynamic_k(self.match_count.get(loser, 0))
        ) / 2.0
        self.overall_elo[winner], self.overall_elo[loser] = update_elo(w_elo, l_elo, 1, k)
        self.surface_elo[surface][winner], self.surface_elo[surface][loser] = update_elo(
            w_selo, l_selo, 1, k
        )
        self.match_count[winner] = self.match_count.get(winner, 0) + 1
        self.match_count[loser] = self.match_count.get(loser, 0) + 1

        for player, pref in [(winner, "w_"), (loser, "l_")]:
            svpt = row.get(pref + "svpt", None)
            if not _is_missing(svpt) and svpt > 0:
                first_in = row.get(pref + "1stIn", 0) or 0
                new_srv = {
                    "ace_rate": (row.get(pref + "ace", 0) or 0) / svpt,
                    "df_rate": (row.get(pref + "df", 0) or 0) / svpt,
                    "first_in": first_in / svpt,
                    "first_won": (row.get(pref + "1stWon", 0) or 0) / max(first_in, 1),
                    "second_won": (row.get(pref + "2ndWon", 0) or 0) / max(svpt - first_in, 1),
                }
                old_srv = self.serve.get(player)
                self.serve[player] = (
                    new_srv
                    if old_srv is None
                    else {
                        kk: EWMA_ALPHA * new_srv[kk] + (1.0 - EWMA_ALPHA) * old_srv[kk]
                        for kk in new_srv
                    }
                )

        for player in (winner, loser):
            self.match_dates.setdefault(player, []).append(match_date)
            self.match_dates[player] = [
                d for d in self.match_dates[player] if (match_date - d).days <= 14
            ]

        key = tuple(sorted([winner, loser]))
        rec = self.h2h.get(key, {})
        rec[winner] = rec.get(winner, 0) + 1
        rec.setdefault(loser, 0)
        self.h2h[key] = rec


def prepare_ml_dataset(matches_df):
    """Build mirrored, chronologically ordered training/evaluation DataFrame and StateTracker."""
    import pandas as pd

    print("Construim dataset-ul (Elo, serviciu, H2H, fatigue) in ordine cronologica...")
    matches_df = matches_df.copy()
    matches_df["tourney_date"] = pd.to_datetime(matches_df["tourney_date"])
    sort_cols = ["tourney_date"] + (
        ["match_num"] if "match_num" in matches_df.columns else []
    )
    matches_df = matches_df.sort_values(sort_cols).reset_index(drop=True)

    st = StateTracker()
    rows = []

    for _, r in matches_df.iterrows():
        if pd.isna(r["winner_name"]) or pd.isna(r["loser_name"]) or pd.isna(r["surface"]):
            continue
        w, l, surf, match_date = r["winner_name"], r["loser_name"], r["surface"], r["tourney_date"]
        if surf not in st.surface_elo:
            continue

        w_rank = 500 if pd.isna(r.get("winner_rank", float("nan"))) else r.get("winner_rank")
        l_rank = 500 if pd.isna(r.get("loser_rank", float("nan"))) else r.get("loser_rank")
        w_age = r.get("winner_age", 26) if not pd.isna(r.get("winner_age", float("nan"))) else 26
        l_age = r.get("loser_age", 26) if not pd.isna(r.get("loser_age", float("nan"))) else 26
        best_of = r.get("best_of", 3)

        w_state = st.get_state(w, surf, match_date, w_rank, w_age)
        l_state = st.get_state(l, surf, match_date, l_rank, l_age)
        w_h2h, l_h2h, total_h2h = st.h2h_record(w, l)
        w_state["h2h_wins"], w_state["h2h_total"] = w_h2h, total_h2h
        l_state["h2h_wins"], l_state["h2h_total"] = l_h2h, total_h2h

        row_w = build_features(w_state, l_state, best_of)
        row_w["target"] = 1
        row_l = build_features(l_state, w_state, best_of)
        row_l["target"] = 0
        for rr, p1n, p2n in [(row_w, w, l), (row_l, l, w)]:
            rr["meta_date"] = match_date
            rr["meta_p1"] = p1n
            rr["meta_p2"] = p2n
        rows.append(row_w)
        rows.append(row_l)

        st.update_after_match(w, l, surf, match_date, r)

    df = pd.DataFrame(rows)
    print(f"Dataset gata: {len(df)} randuri ({len(df) // 2} meciuri x 2 orientari).")
    return df, st
