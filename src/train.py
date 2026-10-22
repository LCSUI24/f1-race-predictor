import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import ndcg_score

DATA_PROCESSED_PATH = os.path.join("data", "processed", "f1_dataset_processed.parquet")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def load_dataset():
    """Memuat dataset yang sudah bersih."""
    if not os.path.exists(DATA_PROCESSED_PATH):
        raise FileNotFoundError(f"File {DATA_PROCESSED_PATH} tidak ditemukan. Jalankan src/features.py terlebih dahulu.")
    df = pd.read_parquet(DATA_PROCESSED_PATH)
    return df


def split_train_test(df: pd.DataFrame, test_year: int = 2023):
    """
    Split data berbasis waktu (temporal split).
    Data sebelum test_year untuk training, test_year untuk pengujian.
    """
    train_df = df[df["year"] < test_year].copy()
    test_df = df[df["year"] == test_year].copy()

    # Pastikan data terurut per balapan agar grup query LTR valid
    train_df.sort_values(by=["year", "round"], inplace=True)
    test_df.sort_values(by=["year", "round"], inplace=True)

    return train_df, test_df


def train_ranking_model():
    print("Memuat dataset olahan...")
    df = load_dataset()

    # Definisikan fitur input
    features = [
        "grid_position",
        "fp2_pace_delta",
        "qual_speed_delta",
    ]

    # Tambahkan metrik cuaca jika tersedia di kolom
    for col in ["fp2_avg_track_temp", "fp2_avg_air_temp"]:
        if col in df.columns:
            features.append(col)
            df[col] = df[col].fillna(df[col].median())

    print(f"Fitur yang digunakan: {features}")

    # Temporal split: Data 2018-2022 untuk train, 2023 untuk test
    train_df, test_df = split_train_test(df, test_year=2023)

    X_train = train_df[features]
    y_train = train_df["rank_target"]
    groups_train = train_df.groupby(["year", "round"]).size().to_numpy()

    X_test = test_df[features]
    y_test = test_df["rank_target"]
    groups_test = test_df.groupby(["year", "round"]).size().to_numpy()

    print(f"Data Train: {len(train_df)} baris ({len(groups_train)} balapan)")
    print(f"Data Test: {len(test_df)} baris ({len(groups_test)} balapan)")

    # Inisialisasi LGBMRanker
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[1, 3, 10],
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42
    )

    print("\nMelatih model LGBMRanker...")
    ranker.fit(
        X_train,
        y_train,
        group=groups_train,
        eval_set=[(X_test, y_test)],
        eval_group=[groups_test],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )

    # -------------------------------------------------------------
    # Evaluasi Prediksi Balapan Musim 2023
    # -------------------------------------------------------------
    test_df["predicted_score"] = ranker.predict(X_test)

    # Urutkan prediksi dalam tiap race (skor tertinggi = P1 prediksi)
    test_df["predicted_rank"] = (
        test_df.groupby(["year", "round"])["predicted_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    total_races = test_df["round"].nunique()
    exact_winner_hits = 0
    podium_precision_list = []

    for _, race_group in test_df.groupby("round"):
        # Juara asli vs prediksi
        actual_winner = race_group[race_group["finish_position"] == 1]["driver_code"].values
        pred_winner = race_group[race_group["predicted_rank"] == 1]["driver_code"].values

        if len(actual_winner) > 0 and len(pred_winner) > 0:
            if actual_winner[0] == pred_winner[0]:
                exact_winner_hits += 1

        # Podium asli (P1-P3) vs podium prediksi
        actual_podium = set(race_group[race_group["finish_position"] <= 3]["driver_code"])
        pred_podium = set(race_group[race_group["predicted_rank"] <= 3]["driver_code"])
        overlap = len(actual_podium.intersection(pred_podium))
        podium_precision_list.append(overlap / 3.0)

    winner_acc = (exact_winner_hits / total_races) * 100
    avg_podium_acc = np.mean(podium_precision_list) * 100

    print("\n================ HASIL EVALUASI MUSIM 2023 ================")
    print(f"Total Balapan Uji       : {total_races}")
    print(f"Akurasi Penebak Juara 1 : {winner_acc:.2f}%")
    print(f"Rata-rata Akurasi Podium: {avg_podium_acc:.2f}%")

    # Feature Importance
    importance_df = pd.DataFrame({
        "feature": features,
        "importance": ranker.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\n--- Feature Importance ---")
    print(importance_df.to_string(index=False))

    # Simpan model
    model_save_path = os.path.join(MODELS_DIR, "lgbm_f1_ranker.pkl")
    joblib.dump(ranker, model_save_path)
    print(f"\nModel berhasil disimpan ke: {model_save_path}")


if __name__ == "__main__":
    train_ranking_model()