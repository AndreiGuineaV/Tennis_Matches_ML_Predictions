"""Merge annual ATP match CSV files into merged_matches.csv using portable relative paths."""

import argparse
import glob
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "atp_matches"
DEFAULT_OUTPUT_FILE = BASE_DIR / "merged_matches.csv"


def load_and_merge_data(folder_path):
    print("Începem încărcarea și îmbinarea fișierelor CSV...")

    folder = Path(folder_path)
    file_pattern = str(folder / "atp_matches_[1-2]*.csv")
    csv_files = sorted(glob.glob(file_pattern))

    if not csv_files:
        print(f"Eroare: Nu am găsit niciun fișier CSV în folderul specificat: {folder}")
        return None

    dataframes = []
    for file_path in csv_files:
        try:
            df_year = pd.read_csv(file_path, low_memory=False)
            dataframes.append(df_year)
            print(f"Încărcat cu succes: {Path(file_path).name}")
        except Exception as exc:
            print(f"Eroare la citirea fișierului {file_path}: {exc}")

    if not dataframes:
        return None

    matches_df = pd.concat(dataframes, ignore_index=True)
    matches_df["tourney_date"] = pd.to_datetime(
        matches_df["tourney_date"], format="%Y%m%d", errors="coerce"
    )
    matches_df = matches_df.sort_values(by=["tourney_date", "match_num"]).reset_index(drop=True)

    print("\n--- Rezumat ---")
    print(f"Număr total de meciuri încărcate: {len(matches_df)}")
    print(
        f"Perioada acoperită: {matches_df['tourney_date'].min().date()} -> "
        f"{matches_df['tourney_date'].max().date()}"
    )

    return matches_df


def main():
    parser = argparse.ArgumentParser(
        description="Merge annual ATP match CSVs into a single chronologically sorted CSV."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing atp_matches_YYYY.csv files (default: ./atp_matches)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output CSV path (default: ./merged_matches.csv)",
    )
    args = parser.parse_args()

    matches_df = load_and_merge_data(args.input_dir)
    if matches_df is not None:
        matches_df.to_csv(args.output, index=False)
        print(f"Salvat în: {args.output}")


if __name__ == "__main__":
    main()