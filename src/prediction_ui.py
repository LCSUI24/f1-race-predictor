import streamlit as st

from race_model import EDITABLE_FEATURES, GRID, PRACTICE, QUALIFYING, apply_scenario, rank_drivers
from race_ui import podium_card


def render_prediction_form(race, dataset, model, baseline, race_key):
    saved = st.session_state.get("scenario_result")
    if saved is not None and saved[0] != race_key:
        st.session_state.pop("scenario_result")

    st.html('<h2 class="section-title prediction-title">Buat prediksi</h2>')
    st.caption(
        "Ubah grid, pace, atau kondisi trek untuk Grand Prix terpilih. "
        "Model memakai riwayat pembalap dan tim dari dataset sebagai dasar."
    )
    number = st.column_config.NumberColumn
    grid = race[["driver_code", "team_name", *EDITABLE_FEATURES]].sort_values(GRID, kind="stable")
    grid = grid.assign(driver_code=grid["driver_code"].astype(str), team_name=grid["team_name"].astype(str))
    grid_limit = max(22, len(grid), int(grid[GRID].dropna().max()) if grid[GRID].notna().any() else 0)

    with st.form(key=f"prediction_{race_key}", border=True):
        weather_column, temperature_column = st.columns(2)
        with weather_column:
            rain = st.selectbox("Kondisi trek untuk prediksi", (None, 0, 1), format_func={
                None: "Ikuti data balapan", 0: "Kering", 1: "Hujan",
            }.get)
        with temperature_column:
            temp = st.number_input(
                "Suhu aspal untuk prediksi (°C)",
                min_value=-10.0,
                max_value=100.0,
                step=.5,
                value=None,
                placeholder="Ikuti data balapan",
            )

        st.markdown("### Grid & pace pembalap")
        st.caption("Edit angka di tabel. Untuk memindahkan pembalap ke grid yang terisi, tukar posisi keduanya.")
        edited = st.data_editor(
            grid,
            key=f"prediction_grid_{race_key}",
            column_config={
                "driver_code": st.column_config.TextColumn("Pembalap", pinned=True, width="small"),
                "team_name": "Tim",
                GRID: number("Grid", min_value=1, max_value=grid_limit, step=1, format="%d"),
                QUALIFYING: number("Δ Kualifikasi (s)", min_value=0.0, format="%.3f"),
                PRACTICE: number("Δ Pace FP2 (s)", min_value=0.0, format="%.3f"),
            },
            disabled=("driver_code", "team_name"),
            hide_index=True,
            width="stretch",
            row_height=39,
            height=min(39 * (len(grid) + 1), 435),
        )
        if st.form_submit_button("Jalankan prediksi", type="primary"):
            st.session_state.pop("scenario_result", None)
            try:
                scenario = apply_scenario(race, edited, temp, rain)
            except ValueError as error:
                st.error(str(error))
            else:
                result = rank_drivers(scenario, dataset, model)
                result["baseline_rank"] = baseline["pred_rank"]
                result["rank_change"] = result["baseline_rank"] - result["pred_rank"]
                st.session_state["scenario_result"] = (race_key, result)

    saved = st.session_state.get("scenario_result")
    if saved is not None:
        render_prediction_results(saved[1])


def render_prediction_results(result):
    st.html('<h2 class="section-title">Hasil prediksi skenario</h2>')
    st.html(podium_card(result.head(3), "Podium skenario", "pred_rank"))
    st.caption("Perubahan posisi dihitung dari prediksi dengan input awal. Angka positif berarti naik.")
    st.dataframe(
        result[["pred_rank", "driver_code", "team_name", GRID, "baseline_rank", "rank_change", "pred_score"]],
        column_config={
            "pred_rank": st.column_config.NumberColumn("Prediksi", format="%d", width="small"),
            "driver_code": st.column_config.TextColumn("Pembalap", pinned=True, width="small"),
            "team_name": "Tim",
            GRID: st.column_config.NumberColumn("Grid", format="%d", width="small"),
            "baseline_rank": st.column_config.NumberColumn("Posisi awal", format="%d", width="small"),
            "rank_change": st.column_config.NumberColumn("Perubahan", format="%+d", width="small"),
            "pred_score": st.column_config.NumberColumn("Skor model", format="%.3f", width="small"),
        },
        hide_index=True,
        width="stretch",
    )
