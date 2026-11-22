from html import escape
from pathlib import Path
from textwrap import wrap

import pandas as pd
import plotly.express as px
import streamlit as st

from circuit_ui import render_track_timing

ANTHEM_SCRIPT = Path(__file__).with_name("anthem.js").read_text(encoding="utf-8")

TEAM_COLORS = {
    "Red Bull Racing": "#578eed",
    "Red Bull": "#578eed",
    "Ferrari": "#ff5460",
    "Mercedes": "#48dec7",
    "McLaren": "#ff963c",
    "Aston Martin": "#47c397",
    "Alpine": "#ff8fc4",
    "Williams": "#71c5ff",
    "RB": "#8da7ff",
    "Racing Bulls": "#8da7ff",
    "AlphaTauri": "#7baac2",
    "Haas F1 Team": "#d2d6db",
    "Kick Sauber": "#70e569",
    "Sauber": "#70e569",
    "Alfa Romeo": "#e35772",
    "Audi": "#fe4d66",
    "Cadillac": "#edc95e",
}


def team_color(team):
    return TEAM_COLORS.get(team, "#aab0bf")


def render_header(demo):
    source = "Data simulasi" if demo else "Dataset balapan"
    source_class = " is-demo" if demo else ""
    st.html(f"""
        <header class="masthead">
            <div class="brand">
                <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span>
                <span class="wordmark">APEX</span>
                <span class="brand-caption">F1 Race Predictor</span>
            </div>
            <span class="data-source{source_class}">{source}</span>
        </header>
    """)


def render_hero(event, year, round_number):
    event_name = str(event["event_name"])
    name, separator, suffix = event_name.rpartition(" Grand Prix")
    if separator and not suffix and name:
        title = f"{escape(name.strip())}<br>Grand Prix"
    else:
        title = escape(event_name)
    temperature = event.get("r_track_temp")
    temp_label = f"{temperature:.1f} °C" if pd.notna(temperature) else "—"
    rain = event.get("r_rain")
    weather = ("Hujan" if rain == 1 else "Kering") if pd.notna(rain) else "—"
    return f"""
        <section class="race-hero" aria-labelledby="race-title">
            <img class="hero-image" src="./app/static/apex-banner.png" alt="" fetchpriority="high"
                 width="800" height="269">
            <div class="hero-content">
                <p class="race-round">Musim {int(year)} <span></span> Ronde {int(round_number):02d}</p>
                <h1 id="race-title">{title}</h1>
                <dl class="track-details">
                    <div><dt>Kondisi trek</dt><dd>{weather}</dd></div>
                    <div><dt>Suhu aspal</dt><dd>{temp_label}</dd></div>
                </dl>
            </div>
        </section>
    """


def render_metrics(race, predicted, actual):
    predicted_winner = predicted.iloc[0]
    winners = actual.loc[actual["finish_position"] == 1]
    actual_winner = winners.iloc[0] if not winners.empty else None
    status, status_color, status_detail = "—", "", "Hasil P1 belum tersedia"
    if actual_winner is not None:
        winner_rank = int(race.loc[race["driver_code"] == actual_winner["driver_code"], "pred_rank"].iloc[0])
        if winner_rank == 1:
            status, status_color = "Tepat", "positive"
        elif winner_rank == 2:
            status, status_color = "Meleset tipis", "caution"
        else:
            status, status_color = "Meleset", "negative"
        status_detail = f"Juara ada di prediksi P{winner_rank}"

    comparison_ready = len(predicted) == 3 and actual["finish_position"].nunique() == 3
    if comparison_ready:
        overlap = len(set(predicted["driver_code"]) & set(actual["driver_code"]))
        segments = "".join(f'<span class="{"filled" if index < overlap else ""}"></span>' for index in range(3))
        overlap_value = f'{overlap}<span class="metric-denominator"> / 3</span>'
        overlap_detail = "Pembalap yang sama"
        overlap_bar = f'<span class="overlap-scale" aria-hidden="true">{segments}</span>'
    else:
        overlap_value, overlap_detail, overlap_bar = "—", "Podium belum lengkap", ""

    actual_driver = escape(str(actual_winner["driver_code"])) if actual_winner is not None else "—"
    actual_team = escape(str(actual_winner["team_name"])) if actual_winner is not None else "Hasil belum tersedia"
    st.html(f"""
        <dl class="race-metrics" aria-label="Ringkasan prediksi balapan">
            <div class="metric">
                <dt>Prediksi juara</dt>
                <dd class="metric-value">{escape(str(predicted_winner["driver_code"]))}</dd>
                <dd class="metric-detail">{escape(str(predicted_winner["team_name"]))}</dd>
            </div>
            <div class="metric">
                <dt>Juara aktual</dt>
                <dd class="metric-value">{actual_driver}</dd>
                <dd class="metric-detail">{actual_team}</dd>
            </div>
            <div class="metric">
                <dt>Ketepatan juara</dt>
                <dd class="metric-value metric-status {status_color}">{status}</dd>
                <dd class="metric-detail">{status_detail}</dd>
            </div>
            <div class="metric">
                <dt>Kecocokan podium</dt>
                <dd class="metric-value">{overlap_value}{overlap_bar}</dd>
                <dd class="metric-detail">{overlap_detail}</dd>
            </div>
        </dl>
    """)


def podium_card(race, title, position_column):
    is_prediction = position_column == "pred_rank"
    cards = []
    for position in (2, 1, 3):
        rows = race.loc[race[position_column] == position]
        driver, team, color = "—", "Belum tersedia", "#aab0bf"
        grid_text, detail = "", ""
        if not rows.empty:
            row = rows.iloc[0]
            driver = escape(str(row["driver_code"]))
            team = escape(str(row["team_name"]))
            color = team_color(row["team_name"])
            grid = row["grid_position"]
            if pd.notna(grid) and grid > 0:
                grid_text = f"Grid P{int(grid)}"
            if is_prediction:
                detail = f'Skor {row["pred_score"]:.3f}'
            elif grid_text:
                change = int(grid) - position
                direction = "Naik" if change > 0 else "Turun"
                detail = f"{direction} {abs(change)}" if change else "Posisi tetap"

        meta = " · ".join(filter(None, (grid_text, detail)))
        cards.append(f"""
            <li class="podium-driver place-{position}" value="{position}" style="--team-color: {color}">
                <span class="podium-position">P{position}</span>
                <div class="podium-face">
                    <strong class="podium-code">{driver}</strong>
                    <span class="podium-team">{team}</span>
                    <span class="podium-detail">{meta}</span>
                </div>
            </li>
        """)

    panel_class = " prediction-panel" if is_prediction else ""
    return f"""
        <article class="podium-panel{panel_class}">
            <h3>{title}</h3>
            <ol class="podium-places" role="list">{"".join(cards)}</ol>
        </article>
    """


def render_podiums(predicted, actual):
    st.html(f"""
        <section aria-labelledby="podium-title">
            <h2 class="section-title" id="podium-title">Perbandingan podium</h2>
            <div class="podium-panels">
                {podium_card(predicted, "Prediksi model", "pred_rank")}
                {podium_card(actual, "Hasil balapan", "finish_position")}
            </div>
        </section>
    """)


def render_anthem(profile):
    if profile is None:
        markup = '<div data-apex-anthem hidden></div>'
    else:
        driver, recording, audio = profile
        title = escape(recording["title"])
        country = escape(recording["country"])
        markup = f"""
            <section class="anthem-bar" data-apex-anthem data-source="./app/static/anthems/{audio.name}"
                     aria-label="Lagu kebangsaan pemenang">
                <div class="anthem-info">
                    <span class="anthem-mark" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></span>
                    <div>
                        <p class="anthem-label">Lagu kebangsaan pemenang</p>
                        <p><strong>{title}</strong> · {escape(driver)} · {country}</p>
                    </div>
                </div>
                <div class="anthem-controls">
                    <span data-anthem-status role="status"></span>
                    <button type="button" data-anthem-toggle>Putar</button>
                </div>
            </section>
        """
    st.html(f"{markup}<script>{ANTHEM_SCRIPT}</script>", unsafe_allow_javascript=True)


def render_standings(race):
    number = st.column_config.NumberColumn
    column_config = {
        "driver_code": st.column_config.TextColumn("Pembalap", width="small", pinned=True),
        "pred_rank": number("Prediksi", width="small", format="%d"),
        "finish_position": number("Aktual", width="small", format="%d"),
        "finish_gap_display": st.column_config.TextColumn("Gap finis", width="small"),
        "team_name": st.column_config.TextColumn("Tim", width="medium"),
        "grid_position": number("Grid", width="small", format="%d"),
        "q_fastest_delta": number("Δ Kualifikasi (s)", width="small", format="%.3f"),
        "fp2_pace_delta": number("Δ Pace FP2 (s)", width="small", format="%.3f"),
        "driver_season_points": number("Poin musim", width="small", format="%d"),
        "pred_score": number("Skor model", width="small", format="%.3f"),
    }
    columns = race.columns.intersection(column_config, sort=False)
    st.dataframe(
        race[columns],
        column_config=column_config,
        column_order=[column for column in column_config if column in columns],
        hide_index=True,
        width="stretch",
        height=min(37 + 37 * len(race), 484),
        row_height=37,
    )


def show_chart(figure, key, height=365):
    figure.update_layout(
        height=height,
        margin={"l": 16, "r": 24, "t": 26, "b": 14},
        font={"family": "system-ui, sans-serif", "color": "#dadde5", "size": 12},
        paper_bgcolor="#17181d",
        plot_bgcolor="#17181d",
        hoverlabel={"bgcolor": "#25262e", "font_color": "#f6f7fa"},
        showlegend=False,
    )
    figure.update_xaxes(automargin=True, gridcolor="#33353e", zerolinecolor="#666975")
    figure.update_yaxes(automargin=True, gridcolor="#33353e", zerolinecolor="#666975")
    st.plotly_chart(
        figure,
        key=key,
        width="stretch",
        theme=None,
        config={"displaylogo": False, "scrollZoom": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )


def render_telemetry(race):
    colors = {team: team_color(team) for team in race["team_name"].dropna().unique()}
    with st.container(key="telemetry_charts"):
        pace_column, score_column = st.columns(2, gap="medium")

        with pace_column:
            with st.container(border=True):
                st.markdown("### Kualifikasi vs. pace FP2")
                st.caption("Makin dekat ke kiri bawah, makin cepat. Pilih titik untuk melihat detail pembalap.")
                if {"q_fastest_delta", "fp2_pace_delta"}.issubset(race.columns):
                    pace = race.dropna(subset=["q_fastest_delta", "fp2_pace_delta"])
                else:
                    pace = race.iloc[0:0]

                if pace.empty:
                    st.info("Data kualifikasi atau FP2 belum tersedia untuk balapan ini.")
                else:
                    pace = pace.assign(
                        label=pace["driver_code"].astype(str).where(pace["pred_rank"] <= 6, "")
                    )
                    scatter = px.scatter(
                        pace,
                        x="q_fastest_delta",
                        y="fp2_pace_delta",
                        text="label",
                        color="team_name",
                        color_discrete_map=colors,
                        template="plotly_dark",
                        custom_data=["driver_code", "team_name"],
                        labels={"q_fastest_delta": "Δ kualifikasi (s)", "fp2_pace_delta": "Δ pace FP2 (s)"},
                    )
                    scatter.update_traces(
                        textposition="top center",
                        marker={"size": 10, "line": {"color": "#17181d", "width": 1}},
                        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                        "Kualifikasi: %{x:.3f} s<br>FP2: %{y:.3f} s<extra></extra>",
                    )
                    show_chart(scatter, "pace-chart")

        with score_column:
            with st.container(border=True):
                st.markdown("### Skor ranking model")
                st.caption("Skor lebih tinggi menempatkan pembalap lebih atas dalam prediksi.")
                top = race.head(10)
                scores = px.bar(
                    top,
                    x="pred_score",
                    y="driver_code",
                    color="team_name",
                    color_discrete_map=colors,
                    custom_data=["team_name", "pred_rank"],
                    orientation="h",
                    template="plotly_dark",
                    labels={"pred_score": "Skor model", "driver_code": ""},
                )
                scores.update_yaxes(
                    categoryorder="array",
                    categoryarray=top["driver_code"].tolist(),
                    autorange="reversed",
                    showgrid=False,
                )
                scores.update_traces(
                    hovertemplate="<b>%{y}</b> · %{customdata[0]}<br>Prediksi P%{customdata[1]}"
                    "<br>Skor: %{x:.3f}<extra></extra>"
                )
                show_chart(scores, "scores-chart")


def render_importance(model):
    features = pd.DataFrame({"feature": model.feature_name_, "importance": model.feature_importances_})
    features = features.nlargest(15, "importance").sort_values("importance", kind="stable")
    features["label"] = features["feature"].str.replace("_", " ").map(
        lambda label: "<br>".join(wrap(label, width=20))
    )
    with st.container(border=True):
        st.markdown("### 15 fitur teratas")
        by_gain = model.importance_type == "gain"
        if by_gain:
            st.caption("Total peningkatan yang dihasilkan setiap fitur saat membagi pohon LightGBM.")
        else:
            st.caption("Seberapa sering setiap fitur dipakai untuk membagi pohon LightGBM.")
        chart = px.bar(
            features,
            x="importance",
            y="label",
            custom_data=["feature"],
            orientation="h",
            template="plotly_dark",
            labels={"importance": "Total gain" if by_gain else "Jumlah split", "label": ""},
        )
        chart.update_traces(
            marker_color="#ec5360",
            hovertemplate="<b>%{customdata[0]}</b><br>%{x:,.2~f}<extra></extra>",
        )
        chart.update_yaxes(showgrid=False)
        show_chart(chart, "feature-chart", height=600)


def render_analysis(race, model, event, timing_source):
    st.html('<h2 class="section-title analysis-title">Analisis balapan</h2>')
    timing_tab, standings_tab, telemetry_tab, model_tab = st.tabs(
        ["Sirkuit & gap", "Klasemen", "Telemetri", "Fitur model"]
    )

    with timing_tab:
        render_track_timing(event, race, timing_source, TEAM_COLORS)
    with standings_tab:
        render_standings(race)
    with telemetry_tab:
        render_telemetry(race)
    with model_tab:
        render_importance(model)
