import json
from pathlib import Path
import re
from unicodedata import normalize

import pandas as pd
import streamlit as st

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "reference"
STATIC_DIR = Path(__file__).with_name("static")
EVENT_ALIASES = {
    "barcelona": "barcelona catalunya",
    "mexico city": "mexican",
    "sao paulo": "brazilian",
}
RESULT_LABELS = {
    "did not start": "DNS",
    "did not finish": "DNF",
    "did not qualify": "DNQ",
    "disqualified": "DSQ",
    "excluded": "DSQ",
    "not classified": "NC",
    "withdrew": "WD",
    "retired": "DNF",
}


def event_key(value):
    plain = normalize("NFKD", value).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-z0-9]+", " ", plain.lower()).strip().removesuffix(" grand prix")
    return EVENT_ALIASES.get(name, name)


@st.cache_data
def load_reference():
    events = pd.read_csv(REFERENCE_DIR / "events.csv")
    events["event_key"] = events["event_name"].map(event_key)
    timings = pd.read_csv(REFERENCE_DIR / "finish_times.csv", dtype={"position_text": str})
    drivers = json.loads((REFERENCE_DIR / "drivers.json").read_text(encoding="utf-8"))
    anthems = json.loads((REFERENCE_DIR / "anthems.json").read_text(encoding="utf-8"))
    return events, timings, drivers, anthems


def find_event(events, year, round_number, name):
    candidates = events.loc[events["year"].eq(year) & events["event_key"].eq(event_key(name))]
    exact = candidates.loc[candidates["round"].eq(round_number)]
    if not exact.empty:
        return exact.iloc[0]
    if len(candidates) == 1:
        return candidates.iloc[0]
    return None


def seconds_label(gap):
    return "0.000 s" if gap == 0 else f"+{gap:.3f} s"


def finish_timing(race, snapshot, demo):
    def optional_field(name):
        return race.get(name, pd.Series(index=race.index, dtype="object"))

    driver = race["driver_code"].astype(str)
    position = pd.to_numeric(race["finish_position"], errors="coerce")
    seconds = pd.to_numeric(optional_field("finish_gap_seconds"), errors="coerce").astype(float)
    laps = pd.to_numeric(optional_field("finish_gap_laps"), errors="coerce").astype(float)
    seconds = seconds.where(seconds.ge(0) & seconds.lt(float("inf")))
    laps = laps.where(laps.ge(1) & laps.mod(1).eq(0))

    raw_status = optional_field("race_status").astype("string").str.strip()
    reported_laps = raw_status.str.extract(r"^\+\s*(\d+)\s+laps?\b", flags=re.I)[0]
    laps = laps.fillna(pd.to_numeric(reported_laps, errors="coerce"))
    status = raw_status.str.lower().map(RESULT_LABELS).astype("string")
    status = status.fillna(raw_status.where(raw_status.isin(RESULT_LABELS.values())))
    other = raw_status.notna() & ~raw_status.str.fullmatch(r"(?i)finished|classified|", na=False)
    status = status.mask(status.isna() & other & laps.isna(), "DNF")

    used_dataset = bool((seconds.notna() & position.gt(1)).any() or laps.notna().any())
    used_snapshot = False
    winners = driver.loc[position.eq(1)]
    official_winners = snapshot.loc[snapshot["finish_position"].eq(1), "driver_code"]
    if not demo and len(winners) == len(official_winners) == 1 and winners.iloc[0] == official_winners.iloc[0]:
        official = snapshot.drop_duplicates("driver_code").set_index("driver_code")
        compatible_order = position.eq(driver.map(official["finish_position"]))
        reference_seconds = driver.map(official["gap_seconds"]).where(compatible_order & status.isna() & laps.isna())
        reference_laps = driver.map(official["gap_laps"]).where(compatible_order & status.isna() & seconds.isna())
        used_snapshot = bool((seconds.isna() & reference_seconds.notna() & position.gt(1)).any()
                             or (laps.isna() & reference_laps.notna()).any())
        seconds = seconds.fillna(reference_seconds)
        laps = laps.fillna(reference_laps)
        reference_status = driver.map(official["position_text"]).where(seconds.isna() & laps.isna())
        status = status.fillna(reference_status.where(reference_status.isin(RESULT_LABELS.values())))

    seconds = seconds.mask(position.eq(1), 0.0).where(laps.isna() & status.isna() & position.ge(1))
    labels = seconds.map(lambda gap: seconds_label(gap) if pd.notna(gap) else "—")
    labels = labels.mask(laps.notna(), laps.map(lambda value: f"+{int(value)} lap" if pd.notna(value) else "—"))
    labels = labels.mask(status.notna(), status)

    if demo:
        source = "Data simulasi"
    elif used_snapshot:
        source = "Dataset balapan dan F1DB" if used_dataset else "F1DB"
    else:
        source = "Dataset balapan"

    return race.assign(
        gap_seconds=seconds,
        gap_laps=laps,
        result_status=status,
        finish_gap_display=labels,
    ), source


def winner_anthem(actual, drivers, anthems):
    winners = actual.loc[actual["finish_position"].eq(1), "driver_code"]
    if winners.empty:
        return None
    code = str(winners.iloc[0])
    driver = drivers.get(code, {})
    country = driver.get("country", "")
    recording = anthems.get(country)
    if recording is None:
        return None
    audio_path = STATIC_DIR / "anthems" / f"{country}.mp3"
    if audio_path.is_file():
        return driver.get("name", code), recording, audio_path
    return None
