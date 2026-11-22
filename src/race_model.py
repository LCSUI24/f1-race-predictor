import math

import numpy as np
import pandas as pd

GRID = "grid_position"
QUALIFYING = "q_fastest_delta"
PRACTICE = "fp2_pace_delta"
EDITABLE_FEATURES = (GRID, QUALIFYING, PRACTICE)


def rank_drivers(race, dataset, model):
    features = list(model.feature_name_)
    inputs = race[features].copy()
    numeric_columns = inputs.select_dtypes(include="number").columns
    inputs[numeric_columns] = inputs[numeric_columns].fillna(dataset[numeric_columns].median())
    result = race.assign(pred_score=model.predict(inputs))
    result["pred_rank"] = result["pred_score"].rank(ascending=False, method="first").astype(int)
    return result.sort_values("pred_rank")


def apply_scenario(race, edited, track_temp, rain):
    scenario = race.copy()
    edited = edited.reindex(scenario.index)
    if track_temp is not None:
        if not math.isfinite(track_temp) or not -10 <= track_temp <= 100:
            raise ValueError("Suhu aspal harus berada di antara −10 dan 100 °C.")
        scenario["r_track_temp"] = float(track_temp)
    if rain is not None:
        scenario["r_rain"] = int(rain)

    changes = {}
    for column in EDITABLE_FEATURES:
        original = scenario[column].astype(float)
        values = pd.to_numeric(edited[column], errors="coerce").astype(float)
        changed = values.ne(original) & (values.notna() | original.notna())
        if (changed & (~np.isfinite(values) | values.lt(0))).any():
            raise ValueError("Isi nilai delta dan grid yang diubah dengan angka nonnegatif.")
        changes[column] = changed

        if column == GRID:
            if (changed & (values.lt(1) | values.mod(1).ne(0))).any():
                raise ValueError("Posisi grid harus berupa angka bulat mulai dari 1.")
            used = values.duplicated(keep=False)
            if (changed & used).any():
                duplicate_positions = sorted(set(values.loc[changed & used].astype(int)))
                positions = ", ".join(f"P{position}" for position in duplicate_positions)
                raise ValueError(f"Grid {positions} dipakai lebih dari satu pembalap. Tukar posisi sebelum menjalankan prediksi.")

        if changed.any():
            scenario[column] = values

    groups = scenario.groupby("team_name", observed=True)
    if changes[GRID].any():
        scenario["team_round_best_grid"] = groups[GRID].transform("min")
        scenario["grid_delta_to_teammate"] = scenario[GRID] - scenario["team_round_best_grid"]
        scenario["grid_vs_car_potential"] = scenario[GRID] - scenario["team_rolling_3_finish"]

    if changes[QUALIFYING].any():
        scenario["team_round_best_quali"] = groups[QUALIFYING].transform("min")
        scenario["quali_delta_to_teammate"] = scenario[QUALIFYING] - scenario["team_round_best_quali"]

    if changes[PRACTICE].any():
        other_practice = [col for col in ("fp1_pace_delta", "fp3_pace_delta") if col in scenario]
        if other_practice:
            alternative = scenario[other_practice].min(axis=1)
        else:
            previous_best = race["combined_fp_pace_delta"]
            alternative = previous_best.where(race[PRACTICE].isna() | previous_best.lt(race[PRACTICE]))
        practice_best = pd.concat([alternative, scenario[PRACTICE]], axis=1).min(axis=1)
        scenario["combined_fp_pace_delta"] = practice_best.where(
            changes[PRACTICE], race["combined_fp_pace_delta"]
        )
        team_best = scenario.groupby("team_name", observed=True)["combined_fp_pace_delta"].transform("min")
        scenario["fp_delta_to_teammate"] = scenario["combined_fp_pace_delta"] - team_best

    return scenario
