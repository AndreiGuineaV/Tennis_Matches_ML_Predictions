# Tennis Match ML Predictions

A Python project for estimating the winner of men's ATP tennis matches. It builds chronological player-form features, trains and calibrates an XGBoost classifier, evaluates predictions on a later period, optionally backtests them against historical betting odds, and can make a prediction for a manually specified upcoming match.

Predictions and backtests are experimental estimates, not guarantees of match outcomes or profit. Do not treat them as betting advice.

## Project contents

| Path | Purpose |
| --- | --- |
| `tennis.py` | Feature engineering, model training and calibration, evaluation, odds matching, backtesting, and the live-prediction example. |
| `merged_matches.csv` | Combined historical ATP match results consumed by `tennis.py`. |
| `atp_matches/` | Annual ATP match-result CSVs, currently named `atp_matches_YYYY.csv`. |
| `matches_data_merge.py` | Combines matching annual CSVs from `atp_matches/` into `merged_matches.csv`. |
| `2026.csv` | Additional 2026 match data used by the seasonal merge helper. |
| `new_matches_data_merge.py` | Adds `2026.csv` to `merged_matches.csv`, removes duplicate tournament/match IDs, and sorts the result. |
| `odds/` | Historical tennis-data.co.uk Excel files used by the backtest. |
| `odds_scraping.py` | Downloads yearly odds files for 2010–2026 into `odds/`. |

`tempCodeRunnerFile.py` is not part of the documented run pipeline.

## Requirements

Use a Python environment with these packages installed:

```powershell
python -m pip install pandas numpy xgboost scikit-learn openpyxl
```

There is no dependency lock or requirements file in the project at present, so package versions are not pinned. `openpyxl` is needed for reading the `.xlsx` odds files.

Run commands from the project root so the relative paths such as `merged_matches.csv` and `odds/` resolve correctly:

```powershell
cd "C:\Users\AndreiGuinea\OneDrive\Documente\Python Projects\Tennis_Matches_ML_Predictions"
```

## Run the full workflow

After preparing `merged_matches.csv` and (optionally) the odds files, run:

```powershell
python tennis.py
```

The script will:

1. Read `merged_matches.csv`.
2. Sort matches chronologically and calculate pre-match features for both player orientations. The features include overall and surface Elo, rank, recent-match fatigue, age, recent serve statistics, head-to-head record, and best-of format.
3. Train XGBoost using the earliest 70% of match rows, use the next 15% for early stopping and probability calibration, and evaluate on the final 15%.
4. Print accuracy, log loss, Brier score, and calibration-bin results for the test period.
5. Find `odds/tennis_odds_*.xlsx`, attach odds where a match can be matched, and run a backtest if usable rows are found.
6. Run the example live prediction at the bottom of `tennis.py`.

The chronological split is made on match pairs so the mirrored rows for one match are kept together. The model and player state are rebuilt each run; the script does not save a trained model or prediction results to disk.

### Get a prediction for a different match

Edit the `predict_match(...)` call near the bottom of `tennis.py` and run the script again. Set the player names, surface, best-of format, and available ranking/age inputs to the values for the upcoming match. The default surface must be one of `Hard`, `Clay`, `Grass`, or `Carpet`.

The function prints the model's estimated win probability for each player and returns player 1's probability. To compare with market odds, supply decimal `odds_p1` and `odds_p2` arguments. The printed market comparison removes the bookmaker overround; an edge is only labelled as value when it exceeds 3 percentage points. Player state comes from the historical matches in `merged_matches.csv`, so keep that file current before making a prediction.

## Update match data

Keep a backup of `merged_matches.csv` before replacing or merging data. The feature pipeline expects Jeff Sackmann-style match columns, including at least `tourney_date`, `winner_name`, `loser_name`, and `surface`. It also uses ranking, age, best-of, match number, and serve-stat columns when available. Dates must be parseable by pandas, and supported surface values are `Hard`, `Clay`, `Grass`, and `Carpet`; rows with missing player names/surface or unsupported surfaces are skipped.

### Rebuild from annual ATP CSVs

1. Put the completed annual result files in `atp_matches/` using names such as `atp_matches_2027.csv`.
2. Open `matches_data_merge.py` and change `folder_path` at the bottom to the full path of this project's `atp_matches` folder. The current value is specific to the original Windows machine.
3. From the project root, run:

   ```powershell
   python matches_data_merge.py
   ```

   This reads files matching `atp_matches_[1-2]*.csv`, sorts them by tournament date and match number, and writes `merged_matches.csv` in the current directory. It does not include `ongoing_tourneys.csv`.

### Add the current 2026 feed

`new_matches_data_merge.py` reads `merged_matches.csv` and `2026.csv`, drops the `indoor` column from the new feed, deduplicates on `tourney_id` and `match_num` (keeping the new-feed row), sorts chronologically, and overwrites `merged_matches.csv`.

Use this helper only when `2026.csv` is the intended current-season feed. Confirm that it does not omit previously merged results, and back up the combined CSV before running it. The script is hard-coded to 2026; update its input and deduplication workflow for later seasons rather than assuming it updates itself.

After either update, check that dates parse correctly, rows are sorted, and duplicate matches have not been introduced before running `tennis.py`.

## Update betting odds (optional)

From the project root, run:

```powershell
python odds_scraping.py
```

The helper attempts to download `.xlsx` files (with a `.zip` fallback) for 2010–2026 from tennis-data.co.uk into `odds/`. It may overwrite files with the same names. Check the script output for failed years and verify the downloaded files before relying on the backtest.

`tennis.py` expects odds data with `Winner`, `Loser`, `Date`, and the bookmaker columns `B365W`, `B365L`, `PSW`, and `PSL`. `B365W`/`B365L` are used as the offered odds and `PSW`/`PSL` as closing odds for the CLV calculation. If using a different odds provider, update the `bet_book` and `close_book` arguments/defaults in `attach_odds()` and confirm that its player-name format and winner/loser orientation match the join logic.

The backtest uses a quarter-Kelly stake fraction by default, caps each stake at 5% of the current bankroll, and requires at least a 3-percentage-point edge over the no-vig market probability. Its output depends on the available and successfully matched odds; unmatched or incomplete odds are excluded.

## Understanding results and limitations

- **Accuracy** is the fraction of test outcomes classified correctly at a 50% probability threshold. It does not measure probability quality on its own.
- **Log loss** and **Brier score** assess probabilistic predictions; lower is better.
- **Calibration output** compares predicted-probability bins with observed win rates.
- **Backtest ROI, drawdown, and CLV** describe the selected historical sample and the odds available to the join. They are not evidence of future profitability.
- Match rows are mirrored so each player appears once as the predicted first player and once as the second player. The odds join/backtest reduces this to one orientation per match.
- The model is trained on match outcomes and a small set of engineered player features. It does not incorporate all potentially relevant information such as injuries, rest quality, tournament context, or live market movement.
- Historical rankings/ages, missing serve statistics, odds coverage, player-name matching, and changes in tour conditions can all affect results.

## What could be improved

1. **Make runs reproducible:** add a `requirements.txt` or `pyproject.toml` with tested dependency versions, and record dataset source/version and run parameters.
2. **Remove machine-specific paths:** make the match-merge script accept a relative/default data directory or command-line argument instead of a hard-coded Windows path.
3. **Automate and validate data updates:** replace the 2026-specific merge helper with a year-agnostic update step; validate required columns, dates, supported surfaces, duplicate IDs, and chronological order before overwriting the main dataset.
4. **Strengthen evaluation:** use multiple rolling-origin test windows, report metrics by season/surface/tournament, and reserve a distinct chronological calibration period rather than reusing the early-stopping validation period for calibration.
5. **Test the pipeline:** add unit tests for Elo/state updates, feature symmetry, odds matching, vig removal, staking, and data leakage/order guarantees.
6. **Improve odds matching:** use stable player identifiers and tournament/match identifiers where available, report match rates by year, and make bookmaker columns/provider configurable.
7. **Persist artifacts:** save the fitted model, calibration object, feature list, training cutoff, and metrics together so predictions can be reproduced without retraining.
8. **Broaden match context carefully:** investigate tournament level, round, indoor status, rest/travel, injuries, and additional rolling performance indicators, ensuring every feature is known before the match.

