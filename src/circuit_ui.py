import json
import math
from html import escape
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import streamlit as st

from race_data import REFERENCE_DIR, STATIC_DIR, seconds_label

MAP_SCRIPT = Path(__file__).with_name("circuit_map.js").read_text(encoding="utf-8")
NEUTRAL_MARKER = "#aab0bf"


@st.cache_data
def circuit_path(layout):
    source = STATIC_DIR / "circuits" / f"{layout}.svg"
    if not source.is_file():
        return None
    path = ElementTree.parse(source).getroot().find("{http://www.w3.org/2000/svg}path")
    return path.get("d") if path is not None else None


@st.cache_data
def track_anchors():
    return json.loads((REFERENCE_DIR / "track_timing.json").read_text(encoding="utf-8"))


def track_gap_scale(race, event):
    # Use the recorded median race lap when supplied with the user's dataset.
    lap_seconds = event.get("winner_lap_seconds", math.nan)
    if "r_median_pace" in race:
        pace = pd.to_numeric(
            race.loc[race["finish_position"].eq(1), "r_median_pace"], errors="coerce"
        ).dropna()
        if not pace.empty and math.isfinite(pace.iloc[0]) and pace.iloc[0] > 0:
            lap_seconds = float(pace.iloc[0])
    if not math.isfinite(lap_seconds) or lap_seconds <= 0:
        lap_seconds = 100.0
    largest_gap = race["gap_seconds"].max()
    # A classified driver's marker should not wrap through the finish line.
    return max(lap_seconds, float(largest_gap) * 1.05) if pd.notna(largest_gap) else lap_seconds


def timing_rows(race, team_colors, on_map):
    ordered = race.sort_values("finish_position", kind="stable")
    timed = ordered.loc[ordered["gap_seconds"].notna()]
    rows = []
    for key, row in enumerate(timed.itertuples()):
        code = escape(str(row.driver_code))
        gap = float(row.gap_seconds)
        position = int(row.finish_position)
        tint = team_colors.get(row.team_name, NEUTRAL_MARKER)
        row_content = f"""
            <span class="gap-pos">P{position:02d}</span>
            <span class="gap-dot" style="--marker-color: {tint}" aria-hidden="true"></span>
            <strong>{code}</strong>
            <span class="gap-value">{seconds_label(gap)}</span>
        """
        if on_map:
            row_content = f"""
                <button type="button" data-driver-row="{key}" data-code="{code}"
                        data-seconds="{gap:.4f}" data-tint="{tint}" aria-pressed="false">
                    {row_content}
                </button>
            """
        else:
            row_content = f'<div class="gap-static-row">{row_content}</div>'
        rows.append(f"<li>{row_content}</li>")

    unplotted = ordered.loc[ordered["gap_seconds"].isna() & ordered["finish_gap_display"].ne("—")]
    other_rows = "".join(
        f"<li><strong>{escape(str(row.driver_code))}</strong> {escape(row.finish_gap_display)}</li>"
        for row in unplotted.itertuples()
    )
    statuses = f'<p>Status finis</p><ul class="gap-statuses" role="list">{other_rows}</ul>' if other_rows else ""
    listing = "".join(rows)
    if listing:
        listing = f'<ol class="gap-drivers" role="list" aria-label="Gap waktu pembalap">{listing}</ol>'
    if not timed["gap_seconds"].gt(0).any():
        listing = f'<p>Catatan gap pembalap lain belum tersedia.</p>{listing}'
    return listing, statuses


def render_track_timing(event, race, timing_source, team_colors):
    layout = event["layout"] if event is not None else None
    path = circuit_path(layout) if pd.notna(layout) else None
    listing, statuses = timing_rows(race, team_colors, path is not None)
    circuit_html = '<div class="circuit-empty">Peta sirkuit belum tersedia untuk balapan ini.</div>'
    title, location, stats, explanation = "Peta sirkuit", "", "", ""
    attributes = ""
    if path is not None:
        anchor = track_anchors().get(layout, {"start": 0, "direction": 1, "calibrated": False})
        label = "Garis finis (perkiraan)" if anchor["calibrated"] else "Titik acuan P1 pada peta"
        scale = track_gap_scale(race, event)
        title = escape(event["circuit_name"])
        location = escape(f"{event['place']}, {event['country']}")
        circuit_html = f"""
            <div class="circuit-diagram" data-track-diagram data-path="{escape(path)}">
                <img src="./app/static/circuits/{escape(layout)}.svg"
                     alt="Denah lintasan {title} dengan titik gap finis" width="500" height="500">
                <p class="circuit-key"><span></span>{label}</p>
            </div>
        """
        attributes = f'data-apex-gap-map data-start="{anchor["start"]}" data-direction="{anchor["direction"]}" data-lap-scale="{scale:.4f}"'
        explanation = f"Posisi titik diperkirakan dari gap waktu. Satu putaran pada peta mewakili {scale:.1f} detik."
        if pd.notna(event["length_km"]):
            stats += f"<div><dt>Panjang lintasan</dt><dd>{event['length_km']:.3f} km</dd></div>"
        if pd.notna(event["turns"]):
            stats += f"<div><dt>Tikungan</dt><dd>{int(event['turns'])}</dd></div>"

    st.html(f"""
        <section class="circuit-card" {attributes} aria-labelledby="circuit-name">
            <div class="circuit-map-view">
                <div class="circuit-header">
                    <div><h3 id="circuit-name">{title}</h3><p>{location}</p></div>
                    <dl class="circuit-facts">{stats}</dl>
                </div>
                {circuit_html}
                <p class="circuit-explanation">{explanation}</p>
            </div>
            <div class="gap-board">
                <div class="gap-heading">
                    <h3>Gap ke pemenang</h3>
                    <p>Selisih saat finis · {escape(timing_source)}</p>
                </div>
                {listing}
                {statuses}
            </div>
        </section>
        <script>{MAP_SCRIPT}</script>
    """, unsafe_allow_javascript=True)
