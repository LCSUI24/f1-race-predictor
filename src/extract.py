import os
import time
import warnings
import fastf1
from fastf1.exceptions import RateLimitExceededError
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 1. Direktori Cache dan Output
CACHE_DIR = "f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

DATA_RAW_DIR = os.path.join("data", "raw")
os.makedirs(DATA_RAW_DIR, exist_ok=True)


def parse_session_safe(year: int, round_num: int, session_type: str, max_retries: int = 5):
    """Memuat data sesi dengan auto-wait jika terkena rate limit."""
    retries = 0
    while retries < max_retries:
        try:
            session = fastf1.get_session(year, round_num, session_type)
            session.load(telemetry=False, weather=True, messages=False)
            return session
        except RateLimitExceededError:
            print(f"\n[RATE LIMIT] Kuota panggilan habis pada {year} R{round_num} {session_type}.")
            print("Menunggu 20 menit agar limit direset...")
            time.sleep(1200)  # Tidur 20 menit (1200 detik)
            retries += 1
        except Exception:
            # Sesi dibatalkan atau tidak tersedia
            return None
    return None


def extract_laps_metrics(session, session_name: str):
    """Mengekstrak data putaran individual dan metrik agregat per pembalap."""
    if session is None:
        return pd.DataFrame(), pd.DataFrame(), np.nan, np.nan, 0

    try:
        if session.laps is None or session.laps.empty:
            return pd.DataFrame(), pd.DataFrame(), np.nan, np.nan, 0
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), np.nan, np.nan, 0

    avg_track_temp = np.nan
    avg_air_temp = np.nan
    rain_occurred = 0

    try:
        if session.weather_data is not None and not session.weather_data.empty:
            avg_track_temp = session.weather_data["TrackTemp"].mean()
            avg_air_temp = session.weather_data["AirTemp"].mean()
            rain_occurred = int(session.weather_data["Rainfall"].any())
    except Exception:
        pass

    laps = session.laps.copy()

    req_cols = [
        "Driver", "Team", "LapNumber", "LapTime", "Sector1Time", 
        "Sector2Time", "Sector3Time", "Compound", "TyreLife", 
        "SpeedST", "IsAccurate"
    ]
    for col in req_cols:
        if col not in laps.columns:
            laps[col] = np.nan

    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
    laps["S1Sec"] = laps["Sector1Time"].dt.total_seconds()
    laps["S2Sec"] = laps["Sector2Time"].dt.total_seconds()
    laps["S3Sec"] = laps["Sector3Time"].dt.total_seconds()
    laps["session_type"] = session_name

    driver_metrics = []
    for driver in laps["Driver"].unique():
        d_laps = laps[laps["Driver"] == driver]
        valid_laps = d_laps[d_laps["IsAccurate"] == True]["LapTimeSec"].dropna()

        fastest_lap = valid_laps.min() if not valid_laps.empty else np.nan
        median_pace = valid_laps.median() if not valid_laps.empty else np.nan
        pace_std = valid_laps.std() if len(valid_laps) >= 3 else np.nan
        top_speed_trap = d_laps["SpeedST"].dropna().max()

        driver_metrics.append({
            "driver_code": driver,
            f"{session_name}_fastest_lap": fastest_lap,
            f"{session_name}_median_pace": median_pace,
            f"{session_name}_pace_std": pace_std,
            f"{session_name}_top_speed": top_speed_trap,
            f"{session_name}_laps_completed": len(valid_laps),
        })

    df_summary = pd.DataFrame(driver_metrics)
    return laps, df_summary, avg_track_temp, avg_air_temp, rain_occurred


def extract_round_data(year: int, round_num: int):
    """Mengekstrak seluruh sesi dalam satu akhir pekan balapan."""
    # Ambil metadata event dengan penanganan rate limit
    while True:
        try:
            event = fastf1.get_event(year, round_num)
            event_name = event["EventName"]
            event_format = event["EventFormat"]
            print(f"-> Memproses {year} R{round_num:02d}: {event_name} ({event_format})")
            break
        except RateLimitExceededError:
            print(f"\n[RATE LIMIT] Terkena limit saat membaca jadwal {year} R{round_num}. Tunggu 20 menit...")
            time.sleep(1200)
        except Exception as e:
            print(f"Error metadata {year} R{round_num}: {e}")
            return None, None

    session_keys = ["FP1", "FP2", "FP3", "Q", "Sprint Shootout", "Sprint", "R"]
    all_laps = []
    summaries = []
    weather_summary = {}

    for s_key in session_keys:
        session = parse_session_safe(year, round_num, s_key)
        clean_key = s_key.lower().replace(" ", "_")
        laps, summary, tr_temp, air_temp, rain = extract_laps_metrics(session, clean_key)

        if not laps.empty:
            laps["year"] = year
            laps["round"] = round_num
            laps["event_name"] = event_name
            all_laps.append(laps)

        if not summary.empty:
            summaries.append(summary)

        weather_summary[f"{clean_key}_track_temp"] = tr_temp
        weather_summary[f"{clean_key}_air_temp"] = air_temp
        weather_summary[f"{clean_key}_rain"] = rain

    if not summaries:
        return None, None

    merged_summary = summaries[0]
    for nxt in summaries[1:]:
        merged_summary = merged_summary.merge(nxt, on="driver_code", how="outer")

    race_sess = parse_session_safe(year, round_num, "R")
    if race_sess is not None and race_sess.results is not None:
        res = race_sess.results[["Abbreviation", "TeamName", "Position", "GridPosition", "Status"]].copy()
        res.rename(columns={
            "Abbreviation": "driver_code",
            "TeamName": "team_name",
            "Position": "finish_position",
            "GridPosition": "grid_position",
            "Status": "race_status"
        }, inplace=True)
        final_summary = res.merge(merged_summary, on="driver_code", how="left")
    else:
        final_summary = merged_summary

    final_summary["year"] = year
    final_summary["round"] = round_num
    final_summary["event_name"] = event_name

    for k, v in weather_summary.items():
        final_summary[k] = v

    df_all_laps = pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()
    return final_summary, df_all_laps


def extract_seasons_batch(start_year: int = 2018, end_year: int = 2025):
    """Menjalankan ekstraksi batch 2018-2025 secara otomatis."""
    for year in range(start_year, end_year + 1):
        out_smry = os.path.join(DATA_RAW_DIR, f"f1_sessions_{year}.parquet")
        out_laps = os.path.join(DATA_RAW_DIR, f"f1_laps_{year}.parquet")

        # Lewati musim jika file sudah ada di folder data/raw
        if os.path.exists(out_smry) and os.path.exists(out_laps):
            print(f"\n[SKIP] Musim {year} sudah lengkap di direktori lokal. Melewati...")
            continue

        print(f"\n==================== MEMULAI MUSIM {year} ====================")
        
        while True:
            try:
                schedule = fastf1.get_event_schedule(year)
                races = schedule[schedule["EventFormat"] != "testing"]
                break
            except RateLimitExceededError:
                print(f"\n[RATE LIMIT] Terkena limit saat membaca jadwal {year}. Menunggu 20 menit...")
                time.sleep(1200)
            except Exception as e:
                print(f"Gagal mengambil jadwal musim {year}: {e}")
                races = None
                break

        if races is None:
            continue

        year_summaries = []
        year_laps = []

        for _, row in races.iterrows():
            r_num = int(row["RoundNumber"])
            smry, laps = extract_round_data(year, r_num)

            if smry is not None and not smry.empty:
                year_summaries.append(smry)
            if laps is not None and not laps.empty:
                year_laps.append(laps)

            # Jeda 3 detik per ronde untuk mengurangi beban rate limit
            time.sleep(3.0)

        # Simpan jika seluruh ronde di musim ini berhasil diproses
        if year_summaries:
            df_year_smry = pd.concat(year_summaries, ignore_index=True)
            df_year_smry.to_parquet(out_smry, index=False)
            print(f"Tersimpan: {out_smry}")

        if year_laps:
            df_year_laps = pd.concat(year_laps, ignore_index=True)
            df_year_laps.to_parquet(out_laps, index=False)
            print(f"Tersimpan: {out_laps}")


if __name__ == "__main__":
    extract_seasons_batch(start_year=2018, end_year=2025)