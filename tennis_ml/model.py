"""XGBoost training, isotonic calibration, evaluation, and live match prediction."""

import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from .betting import remove_vig
from .features import FEATURES, build_features

try:
    from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
except ImportError:
    FrozenEstimator = None  # sklearn <= 1.5 uses cv='prefit'


def train_model(df_ml):
    """Train XGBoost with chronological split, calibrate probabilities, and evaluate."""
    x_data, y_data = df_ml[FEATURES], df_ml["target"]
    n_matches = len(df_ml) // 2
    i1, i2 = int(n_matches * 0.70) * 2, int(n_matches * 0.85) * 2
    x_tr, x_val, x_te = x_data.iloc[:i1], x_data.iloc[i1:i2], x_data.iloc[i2:]
    y_tr, y_val, y_te = y_data.iloc[:i1], y_data.iloc[i1:i2], y_data.iloc[i2:]

    print("\nAntrenam XGBoost (early stopping pe log loss)...")
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=6,
        reg_lambda=1.5,
        early_stopping_rounds=80,
        random_state=42,
    )
    model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)

    print("Calibram probabilitatile (izotonic)...")
    if FrozenEstimator is not None:
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    else:
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(x_val, y_val)

    proba = calibrated.predict_proba(x_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print("\n--- Evaluare pe setul de test ---")
    print(f"Acuratete: {accuracy_score(y_te, pred) * 100:.2f}%")
    print(f"Log loss:  {log_loss(y_te, proba):.4f}")
    print(f"Brier:     {brier_score_loss(y_te, proba):.4f}")
    frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=10)
    print("Calibrare (prezis -> real):")
    for mp, fp in zip(mean_pred, frac_pos):
        print(f"  {mp * 100:4.0f}% -> {fp * 100:4.0f}%")

    df_te = df_ml.iloc[i2:].reset_index(drop=True)
    return calibrated, df_te


def predict_match(
    model,
    st,
    player_1,
    player_2,
    surface,
    best_of=3,
    rank_1=500,
    rank_2=500,
    age_1=26,
    age_2=26,
    odds_p1=None,
    odds_p2=None,
):
    """Estimate win probability for an upcoming match and compare with market odds."""
    today = pd.Timestamp.today()
    a = st.get_state(player_1, surface, today, rank_1, age_1)
    b = st.get_state(player_2, surface, today, rank_2, age_2)
    a_wins, b_wins, total = st.h2h_record(player_1, player_2)
    a["h2h_wins"], a["h2h_total"] = a_wins, total
    feats = pd.DataFrame([build_features(a, b, best_of)])[FEATURES]
    p1 = model.predict_proba(feats)[0, 1]

    print(f"\n--- {surface}: {player_1} vs {player_2} ---")
    print(f"Model: {player_1} {p1 * 100:.1f}% | {player_2} {(1 - p1) * 100:.1f}%")
    print(f"H2H: {player_1} {a_wins}-{b_wins} {player_2}")

    if odds_p1 and odds_p2:
        fair_p1, fair_p2, _ = remove_vig(odds_p1, odds_p2)
        for name, my_p, fair_p, odds in [
            (player_1, p1, fair_p1, odds_p1),
            (player_2, 1 - p1, fair_p2, odds_p2),
        ]:
            edge = my_p - fair_p
            tag = f"VALUE (+{edge * 100:.1f}% edge)" if edge > 0.03 else "fara value"
            print(
                f"  {name}: model {my_p * 100:.1f}% vs piata {fair_p * 100:.1f}% (cota {odds}) -> {tag}"
            )
    return p1
