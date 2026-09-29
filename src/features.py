import os
import glob
import numpy as np
import pandas as pd

DATA_RAW_DIR = os.path.join("data", "raw")
DATA_PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

def load_all_sessions():
    """Menggabungkan seluruh session parquet dari data/raw."""
    files = glob.glob(os.path.join(DATA_RAW_DIR, "f1_sessions_*.parquet"))
    if not files:
        raise FileNotFoundError("Tidak ditemukan file f1_sessions_*.parquet di data/raw/")

    dfs = [pd.read_parquet(f) for f in files]
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan data dan membentuk fitur prediktif."""
    df = df.copy()

    # konversi DNF/unclassified ke nilai 20
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors = "coerce").fillna(20).astype(int)
    df["grid_position"] = pd.to_numeric(df["grid_position"], errors = "coerce").fillna(20).astype(int)

    # delta long run pace fp2 ke pembalap tercepat di sesi yang sama
    if "fp2_median_pace" in df.columns:
        group_keys = ["year", "round"]
        fastest_fp2_per_round = df.groupby(group_keys)["fp2_median_pace"].transform("min")
        df["fp2_pace_delta"] = df["fp2_median_pace"] - fastest_fp2_per_round
        
        # Isi jika driver tidak turun di FP2 dengan nilai penalti moderat
        df["fp2_pace_delta"] = df["fp2_pace_delta"].fillna(df["fp2_pace_delta"].max() or 3.0)

    else:
        df["fp2_pace_delta"] = 0.0

    if "q_top_speed" in df.columns:
        median_speed_round = df.groupby(["year", "round"])["q_top_speed"].transform("median")
        df["qual_speed_delta"] = df["q_top_speed"] - median_speed_round
        df["qual_speed_delta"] = df["qual_speed_delta"].fillna(0.0)

    else:
        df["qual_speed_delta"] = 0.0

    # Target Relevansi untuk Learning-to-Rank (P1 = 20 poin, P20 = 1 poin)
    df["rank_target"] = (21 - df["finish_position"]).clip(lower=1, upper=20)

    # Label Encoding
    df["driver_code"] = df["driver_code"].astype(str)
    df["team_name"] = df["team_name"].astype(str)
    df["event_name"] = df["event_name"].astype(str)

    # Urutkan berdasarkan balapan agar pemotongan data urut waktu
    df.sort_values(by=["year", "round", "grid_position"], inplace=True)
    return df


def run_feature_pipeline():
    print("Memuat dataset mentah 2018-2025...")
    raw_df = load_all_sessions()
    print(f"Total baris mentah: {len(raw_df)}")

    processed_df = engineer_features(raw_df)

    output_path = os.path.join(DATA_PROCESSED_DIR, "f1_dataset_processed.parquet")
    processed_df.to_parquet(output_path, index=False)
    print(f"Dataset siap latih tersimpan di: {output_path}")
    print("Dimensi dataset akhir:", processed_df.shape)


if __name__ == "__main__":
    run_feature_pipeline()