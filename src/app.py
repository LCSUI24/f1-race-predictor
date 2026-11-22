from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from prediction_ui import render_prediction_form
from race_data import find_event, finish_timing, load_reference, winner_anthem
from race_model import rank_drivers
from race_ui import (
    render_analysis,
    render_anthem,
    render_header,
    render_hero,
    render_metrics,
    render_podiums,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "f1_dataset_processed.parquet"
DEMO_PATH = ROOT / "data" / "demo_races.csv"
MODEL_PATH = ROOT / "models" / "lgbm_f1_ranker_v4.pkl"
CATEGORY_COLUMNS = ("driver_code", "team_name", "event_name")

st.set_page_config(page_title="Apex | F1 Race Predictor", layout="wide")


@st.cache_data
def load_data(path: Path, modified_ns: int):
    # Keep cached data in sync when the dataset at this path is replaced.
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    for column in CATEGORY_COLUMNS:
        if column in df:
            df[column] = df[column].astype("category")
    return df


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


def select_race(df):
    with st.container(key="race_filters", border=True):
        st.html('<h2 class="filter-title">Pilih balapan</h2>')
        season_column, event_column = st.columns([1, 3], gap="small")

        seasons = sorted(df["year"].unique(), reverse=True)
        with season_column:
            year = st.selectbox("Musim", seasons)

        season = df.loc[df["year"] == year]
        events = (
            season.sort_values("round")
            .drop_duplicates("round")
            .set_index("round")["event_name"]
            .to_dict()
        )
        with event_column:
            round_number = st.selectbox(
                "Grand Prix",
                list(events),
                format_func=lambda number: f"R{int(number):02d} · {events[number]}",
            )

    return season.loc[season["round"] == round_number].copy(), year, round_number


def main():
    st.html(Path(__file__).with_name("theme.css"))

    demo = not DATA_PATH.is_file()
    render_header(demo)
    if not MODEL_PATH.is_file():
        render_anthem(None)
        st.error("Model LightGBM belum tersedia. Tambahkan `models/lgbm_f1_ranker_v4.pkl` dari proyek.")
        return

    source = DEMO_PATH if demo else DATA_PATH
    if not source.is_file():
        render_anthem(None)
        st.error("Dataset belum tersedia. Tambahkan `data/processed/f1_dataset_processed.parquet`.")
        return

    modified_ns = source.stat().st_mtime_ns
    df = load_data(source, modified_ns)
    model = load_model(MODEL_PATH)
    required = set(model.feature_name_) | {"year", "round", "finish_position", *CATEGORY_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        render_anthem(None)
        st.error(f"Kolom dataset berikut belum tersedia: {', '.join(missing)}.")
        return

    race_catalog = df.dropna(subset=["year", "round", "event_name"])
    if race_catalog.empty:
        render_anthem(None)
        st.info("Dataset belum berisi balapan yang bisa ditampilkan.")
        return

    hero = st.empty()
    race_inputs, year, round_number = select_race(race_catalog)
    hero.html(render_hero(race_inputs.iloc[0], year, round_number))

    race = rank_drivers(race_inputs, df, model)
    events, timings, drivers, anthems = load_reference()
    event = find_event(events, year, round_number, str(race["event_name"].iloc[0]))
    snapshot = timings.iloc[0:0]
    if event is not None and not demo:
        snapshot = timings.loc[timings["year"].eq(event["year"]) & timings["round"].eq(event["round"])]
    race, timing_source = finish_timing(race, snapshot, demo)

    predicted = race.head(3)
    actual = race.loc[race["finish_position"].isin([1, 2, 3])].sort_values("finish_position")
    render_anthem(winner_anthem(actual, drivers, anthems))

    with st.container(key="race_views"):
        summary_tab, prediction_tab = st.tabs(["Ringkasan", "Buat prediksi"])
        with summary_tab:
            render_metrics(race, predicted, actual)
            render_podiums(predicted, actual)
            render_analysis(race, model, event, timing_source)

            if demo:
                with st.expander("Gunakan dataset balapan sendiri"):
                    st.markdown(
                        "Mode demo memakai data simulasi. Untuk membuka hasil dari dataset sendiri, "
                        "simpan file di `data/processed/f1_dataset_processed.parquet`, lalu muat ulang "
                        "halaman. Model akan langsung menggunakan file tersebut."
                    )
        with prediction_tab:
            race_key = f"{source.stem}_{modified_ns}_{int(year)}_{int(round_number)}"
            render_prediction_form(race_inputs, df, model, race, race_key)


if __name__ == "__main__":
    main()
