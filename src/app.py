import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="F1 Race Winner & Podium Predictor",
    page_icon="🏎️",
    layout="wide"
)

DATA_PATH = os.path.join("data", "processed", "f1_dataset_processed.parquet")
MODEL_PATH = os.path.join("models", "lgbm_f1_ranker_v4.pkl")

FEATURES = [
    "grid_position", "q_fastest_delta", "quali_delta_to_teammate",
    "grid_delta_to_teammate", "fp_delta_to_teammate", "team_round_best_quali",
    "team_round_best_grid", "combined_fp_pace_delta", "fp2_pace_delta",
    "fp2_consistency", "sprint_pace_delta", "q_speed_delta", "fp2_speed_delta",
    "driver_season_points", "driver_season_wins", "championship_points_ratio",
    "driver_rolling_3_finish", "driver_rolling_5_finish", "driver_rolling_points",
    "driver_win_rate_3", "driver_rolling_grid", "driver_dnf_rate",
    "team_rolling_3_finish", "team_rolling_3_points", "grid_vs_car_potential",
    "driver_track_avg_finish", "fp2_track_temp", "r_track_temp", "r_rain",
    "driver_code", "team_name", "event_name"
]


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    for col in ["driver_code", "team_name", "event_name"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def main():
    st.title("🏎️ F1 Race Outcome & Podium Predictor")
    st.markdown(
        "Prediksi peringkat balapan menggunakan **LightGBM Ranker (LambdaRank)** "
        "berdasarkan telemetri latihan bebas, kualifikasi, cuaca, dan momentum kejuaraan."
    )

    if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
        st.error("File dataset atau model tidak ditemukan. Pastikan sudah menjalankan `features.py` dan `train.py`.")
        return

    df = load_data()
    model = load_model()
    valid_features = [f for f in FEATURES if f in df.columns]

    # --- Sidebar Filter ---
    st.sidebar.header("Pilih Seri Balapan")
    seasons = sorted(df["year"].unique(), reverse=True)
    selected_year = st.sidebar.selectbox("Musim (Tahun)", seasons, index=0)

    season_df = df[df["year"] == selected_year]
    rounds = (
        season_df[["round", "event_name"]]
        .drop_duplicates()
        .sort_values("round")
    )
    round_options = {
        f"R{int(row['round']):02d} - {row['event_name']}": int(row["round"])
        for _, row in rounds.iterrows()
    }
    selected_round_label = st.sidebar.selectbox("Grand Prix", list(round_options.keys()))
    selected_round = round_options[selected_round_label]

    # Filter data untuk seri yang dipilih
    race_df = season_df[season_df["round"] == selected_round].copy()
    for col in valid_features:
        if race_df[col].dtype.name != "category":
            race_df[col] = race_df[col].fillna(df[col].median())

    # Prediksi
    race_df["pred_score"] = model.predict(race_df[valid_features])
    race_df["pred_rank"] = (
        race_df["pred_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    race_df.sort_values("pred_rank", inplace=True)

    # --- Kartu Podium Prediksi vs Aktual ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔮 Prediksi Podium (Model)")
        podium_pred = race_df.head(3)
        p_cols = st.columns(3)
        medals = ["P1", "P2", "P3"]
        for idx, (_, row) in enumerate(podium_pred.iterrows()):
            with p_cols[idx]:
                st.metric(
                    label=f"{medals[idx]} ({row['team_name']})",
                    value=str(row["driver_code"]),
                    delta=f"Grid: P{row['grid_position']}"
                )

    with col2:
        st.subheader("Podium Aktual (Hasil Resmi)")
        podium_actual = race_df.sort_values("finish_position").head(3)
        a_cols = st.columns(3)
        for idx, (_, row) in enumerate(podium_actual.iterrows()):
            with a_cols[idx]:
                diff = int(row["grid_position"]) - int(row["finish_position"])
                st.metric(
                    label=f"{medals[idx]} ({row['team_name']})",
                    value=str(row["driver_code"]),
                    delta=f"{diff:+d} pos dari Grid P{row['grid_position']}"
                )

    st.divider()

    # --- Tabel Perbandingan Lengkap & Grafik ---
    tab1, tab2, tab3 = st.tabs([
        "Tabel Hasil & Prediksi",
        "Analisis Pace & Telemetri",
        "Feature Importance Model"
    ])

    with tab1:
        display_cols = {
            "pred_rank": "Prediksi Pos",
            "finish_position": "Aktual Pos",
            "driver_code": "Pembalap",
            "team_name": "Tim",
            "grid_position": "Grid Start",
            "q_fastest_delta": "Delta Quali (s)",
            "fp2_pace_delta": "Delta Pace FP2 (s)",
            "driver_season_points": "Poin Klasemen",
            "pred_score": "Skor Ranking"
        }
        available_disp = [c for c in display_cols.keys() if c in race_df.columns]
        table_df = race_df[available_disp].rename(columns=display_cols)
        st.dataframe(table_df.style.format(precision=3), use_container_width=True, hide_index=True)

    with tab2:
        c_left, c_right = st.columns(2)
        with c_left:
            fig_pace = px.scatter(
                race_df,
                x="q_fastest_delta",
                y="fp2_pace_delta",
                text="driver_code",
                color="team_name",
                size="driver_season_points",
                title="Kualifikasi Speed vs FP2 Long-Run Pace (Kiri-Bawah = Tercepat)",
                labels={
                    "q_fastest_delta": "Selisih Waktu Kualifikasi ke Pole (detik)",
                    "fp2_pace_delta": "Selisih Median Pace FP2 (detik)"
                }
            )
            fig_pace.update_traces(textposition="top center")
            st.plotly_chart(fig_pace, use_container_width=True)

        with c_right:
            fig_score = px.bar(
                race_df.head(10),
                x="pred_score",
                y="driver_code",
                color="team_name",
                orientation="h",
                title="Top 10 Skor Probabilitas Kemenangan (Output LGBMRanker)",
                labels={"pred_score": "Skor Model", "driver_code": "Pembalap"}
            )
            fig_score.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_score, use_container_width=True)

    with tab3:
        imp_df = pd.DataFrame({
            "Fitur": valid_features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True).tail(15)

        fig_imp = px.bar(
            imp_df,
            x="Importance",
            y="Fitur",
            orientation="h",
            title="Top 15 Fitur Penentu Prediksi Model (LightGBM Split Importance)",
            color="Importance",
            color_continuous_scale="Magma"
        )
        st.plotly_chart(fig_imp, use_container_width=True)


if __name__ == "__main__":
    main()