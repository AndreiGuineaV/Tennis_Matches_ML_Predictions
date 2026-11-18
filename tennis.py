"""Top-level CLI runner for the ATP Tennis Match ML Predictions pipeline."""

import glob
from pathlib import Path

import pandas as pd

from tennis_ml.backtest import attach_odds, backtest
from tennis_ml.features import prepare_ml_dataset
from tennis_ml.model import predict_match, train_model

BASE_DIR = Path(__file__).resolve().parent


def main():
    matches_path = BASE_DIR / "merged_matches.csv"
    matches_df = pd.read_csv(matches_path, low_memory=False)
    df_ml, state = prepare_ml_dataset(matches_df)
    model, df_te = train_model(df_ml)

    odds_files = sorted(glob.glob(str(BASE_DIR / "odds" / "tennis_odds_*.xlsx")))
    if odds_files:
        test_df = attach_odds(df_te, odds_files)
        if not test_df.empty:
            backtest(test_df, model)

    predict_match(
        model,
        state,
        player_1="Zizou Bergs",
        player_2="Toby Samuel",
        surface="Grass",
        best_of=3,
        rank_1=48,
        rank_2=142,
        odds_p1=1.37,
        odds_p2=3.10,
    )


if __name__ == "__main__":
    main()