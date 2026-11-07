import os
import glob
import numpy as np
import pandas as pd

DATA_RAW_DIR = os.path.join("data", "raw")
DATA_PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)


def load_all_sessions():
    files = sorted(glob.glob(os.path.join(DATA_RAW_DIR, "f1_sessions_*.parquet")))
    if not files:
        raise FileNotFoundError("Tidak ditemukan file f1_sessions_*.parquet di data/raw/")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Bersihkan posisi finish & grid
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce").fillna(20).astype(int)
    df["grid_position"] = pd.to_numeric(df["grid_position"], errors="coerce").fillna(20).astype(int)
    df.loc[df["grid_position"] <= 0, "grid_position"] = 20

    # Urutkan secara kronologis murni
    df.sort_values(by=["year", "round", "grid_position"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2. Era Regulasi (0 = 2018-2021, 1 = 2022-2025 Ground Effect)
    df["is_modern_era"] = (df["year"] >= 2022).astype(int)

    # 3. Delta Pace & Fastest Lap di Semua Sesi
    group_keys = ["year", "round"]
    for sess in ["fp1", "fp2", "fp3", "q", "sprint"]:
        pace_col = f"{sess}_median_pace"
        fast_col = f"{sess}_fastest_lap"
        std_col = f"{sess}_pace_std"

        if pace_col in df.columns:
            min_pace = df.groupby(group_keys)[pace_col].transform("min")
            df[f"{sess}_pace_delta"] = (df[pace_col] - min_pace).fillna(2.5)
        else:
            df[f"{sess}_pace_delta"] = 2.5

        if fast_col in df.columns:
            min_fast = df.groupby(group_keys)[fast_col].transform("min")
            df[f"{sess}_fastest_delta"] = (df[fast_col] - min_fast).fillna(2.5)
        else:
            df[f"{sess}_fastest_delta"] = 2.5

        if std_col in df.columns:
            df[f"{sess}_consistency"] = df[std_col].fillna(df[std_col].median())

    # Gabungan Pace Latihan Terbaik (Rata-rata FP1, FP2, FP3 untuk meredam noise cuaca/trafik)
    df["combined_fp_pace_delta"] = df[["fp1_pace_delta", "fp2_pace_delta", "fp3_pace_delta"]].min(axis=1)

    # 4. Proxy Kekuatan Mesin (Speed Trap)
    for sess in ["q", "fp2"]:
        st_col = f"{sess}_top_speed"
        if st_col in df.columns:
            median_st = df.groupby(group_keys)[st_col].transform("median")
            df[f"{sess}_speed_delta"] = (df[st_col] - median_st).fillna(0.0)
        else:
            df[f"{sess}_speed_delta"] = 0.0

    # 5. Indikator Kemenangan, Podium, DNF, dan Poin
    df["is_win"] = (df["finish_position"] == 1).astype(int)
    df["is_podium"] = (df["finish_position"] <= 3).astype(int)
    df["is_dnf"] = (~df["race_status"].astype(str).str.contains("Finished|\\+\\d+ Lap", regex=True)).astype(int)

    f1_points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    df["race_points"] = df["finish_position"].map(f1_points_map).fillna(0.0)

    # 6. KLASEMEN KEJUARAAN MUSIM BERJALAN (Sebelum balapan dimulai)
    df["driver_season_points"] = (
        df.groupby(["year", "driver_code"])["race_points"]
        .transform(lambda x: x.shift(1).cumsum())
        .fillna(0.0)
    )
    df["driver_season_wins"] = (
        df.groupby(["year", "driver_code"])["is_win"]
        .transform(lambda x: x.shift(1).cumsum())
        .fillna(0.0)
    )
    # Rasio poin pembalap terhadap pemimpin klasemen sementara di seri tersebut
    max_season_pts = df.groupby(["year", "round"])["driver_season_points"].transform("max").replace(0, 1)
    df["championship_points_ratio"] = df["driver_season_points"] / max_season_pts

    # 7. FITUR MOMENTUM JANGKA PENDEK (3 & 5 Balapan Terakhir)
    df["driver_rolling_3_finish"] = (
        df.groupby("driver_code")["finish_position"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(12.0)
    )
    df["driver_rolling_5_finish"] = (
        df.groupby("driver_code")["finish_position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(12.0)
    )
    df["driver_rolling_points"] = (
        df.groupby("driver_code")["race_points"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0.0)
    )
    df["driver_win_rate_3"] = (
        df.groupby("driver_code")["is_win"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0.0)
    )
    df["driver_rolling_grid"] = (
        df.groupby("driver_code")["grid_position"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(12.0)
    )
    df["driver_dnf_rate"] = (
        df.groupby("driver_code")["is_dnf"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.1)
    )

    # 8. KEKUATAN TIM & DUEL REKAN SETIM (Teammate Battle)
    team_race = (
        df.groupby(["year", "round", "team_name"], as_index=False)
        .agg(
            team_round_finish_mean=("finish_position", "mean"),
            team_round_points_sum=("race_points", "sum"),
            team_round_best_quali=("q_fastest_delta", "min"),
            team_round_best_grid=("grid_position", "min"),
            team_round_best_fp=("combined_fp_pace_delta", "min")
        )
        .sort_values(by=["year", "round"])
    )

    team_race["team_rolling_3_finish"] = (
        team_race.groupby("team_name")["team_round_finish_mean"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(10.5)
    )
    team_race["team_rolling_3_points"] = (
        team_race.groupby("team_name")["team_round_points_sum"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(2.0)
    )

    df = df.merge(
        team_race[[
            "year", "round", "team_name", 
            "team_rolling_3_finish", "team_rolling_3_points", 
            "team_round_best_quali", "team_round_best_grid", "team_round_best_fp"
        ]],
        on=["year", "round", "team_name"],
        how="left"
    )

    # Selisih langsung dengan rekan setim pada akhir pekan yang sama
    df["quali_delta_to_teammate"] = df["q_fastest_delta"] - df["team_round_best_quali"]
    df["grid_delta_to_teammate"] = df["grid_position"] - df["team_round_best_grid"]
    df["fp_delta_to_teammate"] = df["combined_fp_pace_delta"] - df["team_round_best_fp"]

    # Perkiraan keupayaan menyalip dari posisi grid dibanding kecepatan mobil
    df["grid_vs_car_potential"] = df["grid_position"] - df["team_rolling_3_finish"]

    # 9. Riwayat Spesifik Sirkuit
    df["driver_track_avg_finish"] = (
        df.groupby(["driver_code", "event_name"])["finish_position"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(df["driver_rolling_5_finish"])
    )

    # 10. Target Relevansi (Fokus Tajam di Juara 1 & Podium)
    relevance_map = {
        1: 30, 2: 20, 3: 15, 4: 11, 5: 8, 
        6: 6, 7: 4, 8: 3, 9: 2, 10: 1
    }
    df["rank_target"] = df["finish_position"].map(relevance_map).fillna(0).astype(int)

    for cat_col in ["driver_code", "team_name", "event_name"]:
        df[cat_col] = df[cat_col].astype("category")

    return df


def run_feature_pipeline():
    raw_df = load_all_sessions()
    processed_df = engineer_features(raw_df)
    output_path = os.path.join(DATA_PROCESSED_DIR, "f1_dataset_processed.parquet")
    processed_df.to_parquet(output_path, index=False)
    print(f"Dataset V4 berhasil disimpan ({processed_df.shape[1]} kolom).")


if __name__ == "__main__":
    run_feature_pipeline()