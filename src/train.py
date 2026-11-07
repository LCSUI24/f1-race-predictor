import os
import joblib
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

DATA_PROCESSED_PATH = os.path.join("data", "processed", "f1_dataset_processed.parquet")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def load_dataset():
    return pd.read_parquet(DATA_PROCESSED_PATH)


def build_ranker():
    custom_label_gain = [i for i in range(31)]
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1, 3],
        label_gain=custom_label_gain,
        n_estimators=140,
        learning_rate=0.035,
        max_depth=4,
        num_leaves=12,
        min_child_samples=18,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.5,
        reg_lambda=2.5,
        random_state=42,
        verbose=-1
    )


def evaluate_walk_forward(df: pd.DataFrame, features: list, target_year: int):
    """
    Melatih ulang model setiap sebelum seri dimulai menggunakan seluruh data historis
    yang tersedia hingga seri sebelumnya (Walk-Forward Real-World Simulation).
    """
    season_races = sorted(df[df["year"] == target_year]["round"].unique())
    exact_winner_hits = 0
    top2_winner_hits = 0
    podium_precision_list = []

    print(f"\n================ SIMULASI WALK-FORWARD MUSIM {target_year} ================")
    for r_num in season_races:
        # Data Train: Semua balapan sebelum (target_year, r_num)
        # Beri bobot utama pada era modern (2022+) jika memprediksi 2023-2025
        train_mask = (df["year"] < target_year) | ((df["year"] == target_year) & (df["round"] < r_num))
        if target_year >= 2023:
            # Fokuskan memori pada regulasi ground-effect (2021-sekarang) agar tidak bias era Hamilton 2018-2020
            train_mask = train_mask & (df["year"] >= 2021)

        train_sub = df[train_mask].copy().sort_values(by=["year", "round"])
        test_sub = df[(df["year"] == target_year) & (df["round"] == r_num)].copy()

        X_tr = train_sub[features]
        y_tr = train_sub["rank_target"]
        grp_tr = train_sub.groupby(["year", "round"]).size().to_numpy()

        X_te = test_sub[features]

        model = build_ranker()
        model.fit(X_tr, y_tr, group=grp_tr)

        test_sub["predicted_score"] = model.predict(X_te)
        test_sub["predicted_rank"] = (
            test_sub["predicted_score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )

        event_title = str(test_sub["event_name"].iloc[0])
        actual_w = test_sub[test_sub["finish_position"] == 1]["driver_code"].astype(str).values
        pred_top2 = test_sub.sort_values("predicted_rank")["driver_code"].astype(str).values[:2]

        act_str = actual_w[0] if len(actual_w) > 0 else "N/A"
        prd_str = pred_top2[0] if len(pred_top2) > 0 else "N/A"
        prd_2nd = pred_top2[1] if len(pred_top2) > 1 else "N/A"

        if act_str == prd_str:
            exact_winner_hits += 1
            top2_winner_hits += 1
            status = "BENAR (P1)"
        elif act_str == prd_2nd:
            top2_winner_hits += 1
            status = f"MELESET TIPIS (Juara ada di Prediksi P2: {prd_2nd})"
        else:
            status = "SALAH"

        print(f"{target_year} R{r_num:02d} ({event_title[:18]:<18}) | Pred: {prd_str} (P2:{prd_2nd}) | Aktual: {act_str} -> {status}")

        actual_podium = set(test_sub[test_sub["finish_position"] <= 3]["driver_code"].astype(str))
        pred_podium = set(test_sub[test_sub["predicted_rank"] <= 3]["driver_code"].astype(str))
        podium_precision_list.append(len(actual_podium.intersection(pred_podium)) / 3.0)

    total_races = len(season_races)
    win_acc = (exact_winner_hits / total_races) * 100
    top2_acc = (top2_winner_hits / total_races) * 100
    pod_acc = np.mean(podium_precision_list) * 100

    print(f"\n--- Hasil Akhir Musim {target_year} ---")
    print(f"Akurasi Tepat Juara 1 (Top-1)      : {win_acc:.2f}% ({exact_winner_hits}/{total_races} seri)")
    print(f"Juara Masuk Kandidat Top-2 Prediksi: {top2_acc:.2f}% ({top2_winner_hits}/{total_races} seri)")
    print(f"Rata-rata Presisi Podium (Top-3)   : {pod_acc:.2f}%")
    return model


def main():
    df = load_dataset()

    candidate_features = [
        "grid_position",
        "q_fastest_delta",
        "quali_delta_to_teammate",
        "grid_delta_to_teammate",
        "fp_delta_to_teammate",
        "team_round_best_quali",
        "team_round_best_grid",
        "combined_fp_pace_delta",
        "fp2_pace_delta",
        "fp2_consistency",
        "sprint_pace_delta",
        "q_speed_delta",
        "fp2_speed_delta",
        "driver_season_points",
        "driver_season_wins",
        "championship_points_ratio",
        "driver_rolling_3_finish",
        "driver_rolling_5_finish",
        "driver_rolling_points",
        "driver_win_rate_3",
        "driver_rolling_grid",
        "driver_dnf_rate",
        "team_rolling_3_finish",
        "team_rolling_3_points",
        "grid_vs_car_potential",
        "driver_track_avg_finish",
        "fp2_track_temp",
        "r_track_temp",
        "r_rain",
        "driver_code",
        "team_name",
        "event_name",
    ]

    features = [f for f in candidate_features if f in df.columns]
    for col in features:
        if df[col].dtype.name != "category":
            df[col] = df[col].fillna(df[col].median())

    # Uji Walk-Forward pada musim 2023, 2024, dan 2025
    for yr in [2023, 2024, 2025]:
        if yr in df["year"].values:
            last_model = evaluate_walk_forward(df, features, target_year=yr)

    # Tampilkan Feature Importance dari model terakhir (akhir 2025)
    importance_df = pd.DataFrame({
        "feature": features,
        "importance": last_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\n--- Top 15 Feature Importance (Model Akhir 2025) ---")
    print(importance_df.head(15).to_string(index=False))

    model_save_path = os.path.join(MODELS_DIR, "lgbm_f1_ranker_v4.pkl")
    joblib.dump(last_model, model_save_path)
    print(f"\nModel final disimpan di: {model_save_path}")


if __name__ == "__main__":
    main()