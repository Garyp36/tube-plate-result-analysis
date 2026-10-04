"""Checks for the calculation core. Run with:  pytest -q"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import radiator_core as rc  # noqa: E402


@pytest.fixture(scope="module")
def default_run():
    levels = {k: list(v["default"]) for k, v in rc.FACTOR_SPECS.items()}
    fixed = {k: v[1] for k, v in rc.FIXED_DEFAULTS.items()}
    df, codes, invalid, failures = rc.run_model(levels, fixed)
    return levels, fixed, df, codes, invalid, failures


def test_design_is_balanced_and_orthogonal():
    codes = rc.build_design()
    assert codes.shape == (50, 9)
    for j in range(9):
        assert sorted(np.bincount(codes[:, j])) == [10] * 5


def test_default_inputs_give_50_valid_runs(default_run):
    levels, fixed, df, codes, invalid, failures = default_run
    assert rc.validate_levels(levels) == ([], [])
    assert rc.validate_fixed(fixed) == []
    assert invalid == [] and failures == []
    assert len(df) == 50


def test_energy_balance_closes(default_run):
    df = default_run[2]
    assert df["energy_balance_error_W"].abs().max() < 1e-6


def test_outputs_are_physical(default_run):
    df = default_run[2]
    assert ((df["Effectiveness"] > 0) & (df["Effectiveness"] < 1)).all()
    assert (df["Coolant_Outlet_Temperature_K"] < df["water_temperature_K"]).all()
    assert (df["Q_W"] > 0).all()
    assert (df["Rew"] < 2300).all()


def test_bad_levels_are_rejected(default_run):
    levels = {k: list(v) for k, v in default_run[0].items()}
    levels["water_temperature_K"] = [353.15, 358.15, 363.15, 368.15, 400.0]
    levels["fin_width_m"] = [0.02, 0.016, 0.024, 0.028, 0.032]
    errors, _ = rc.validate_levels(levels)
    assert any("property tables" in e for e in errors)
    assert any("increasing order" in e for e in errors)


def test_high_water_velocity_is_reported_not_crashed(default_run):
    levels = {k: list(v) for k, v in default_run[0].items()}
    levels["water_velocity_m_s"] = [0.05, 0.1, 0.2, 0.4, 0.8]
    df, _, invalid, failures = rc.run_model(levels, default_run[1])
    assert df is None and (invalid or failures)


def test_anova_shares_add_up(default_run):
    df, codes = default_run[2], default_run[3]
    for key in rc.NUMERICAL_RESPONSE_KEYS:
        table, meta = rc.main_effects_anova(codes, df[key].to_numpy())
        assert table["Contribution (%)"].sum() + meta["residual_pct"] == pytest.approx(100.0)
        assert table["Rank"].tolist() == list(range(1, 10))


def test_gra_grades_stay_between_zero_and_one(default_run):
    df = default_run[2]
    keys = ["Coolant_Outlet_Temperature_K", "Q_W"]
    goals = {"Coolant_Outlet_Temperature_K": ("lower", None), "Q_W": ("higher", None)}
    grades, _ = rc.calculate_gra(df, keys, goals, {k: 0.5 for k in keys})
    assert ((grades > 0) & (grades <= 1)).all()
    best = int(np.argmax(grades))
    assert df.loc[best, "Q_W"] > df["Q_W"].median()


def test_target_goal_prefers_values_near_target(default_run):
    df = default_run[2]
    norm = rc.normalize_for_gra(df["Q_W"], ("target", 300.0))
    assert df.loc[int(np.argmax(norm)), "Q_W"] == pytest.approx(
        df.loc[(df["Q_W"] - 300).abs().idxmin(), "Q_W"])


def test_compare_metrics_known_values():
    m = rc.compare_metrics([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    assert m["bias"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(1.0)
    assert m["r"] == pytest.approx(1.0)
    assert rc.agreement_rating(4.9) == "Good"
    assert rc.agreement_rating(10) == "Moderate"
    assert rc.agreement_rating(20) == "Poor"


def test_parse_external_accepts_aliases_and_sorts_by_run(default_run):
    df = default_run[2]
    raw = pd.DataFrame({"Run No": list(range(50, 0, -1)),
                        "Tco (K)": df["Coolant_Outlet_Temperature_K"].to_numpy()[::-1],
                        "q": df["Q_W"].to_numpy()[::-1],
                        "Air dP": np.arange(50.0)})
    out, notes = rc.parse_external(raw)
    assert out["DOE_Run"].tolist() == list(range(1, 51))
    assert out["Q_W"].to_numpy() == pytest.approx(df["Q_W"].to_numpy())
    assert "Air dP" in out.columns


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.iloc[:49], "Expected 50 rows"),
    (lambda d: d.assign(Coolant_Outlet_Temperature_K=d["Coolant_Outlet_Temperature_K"] - 273.15), "Celsius"),
    (lambda d: d.assign(Q_W=d["Q_W"].astype(object).where(d["Q_W"] > 0, "n/a").replace(
        {d["Q_W"].iloc[3]: "oops"})), "non-numeric"),
])
def test_parse_external_rejects_bad_files(default_run, mutate, message):
    good = rc.demo_dataset(default_run[2])
    with pytest.raises(ValueError, match=message):
        rc.parse_external(mutate(good))


def test_empty_run_sheet_gives_a_clear_message(default_run):
    with pytest.raises(ValueError, match="still empty"):
        rc.parse_external(rc.run_sheet(default_run[2]))


def test_demo_data_is_energy_consistent(default_run):
    df = default_run[2]
    demo = rc.demo_dataset(df)
    q_from_tco = (df["water_temperature_K"] - demo["Coolant_Outlet_Temperature_K"]) * df["Cw_W_K"]
    assert q_from_tco.to_numpy() == pytest.approx(demo["Q_W"].to_numpy(), rel=1e-9)


def test_full_analysis_and_export(default_run):
    df, codes = default_run[2], default_run[3]
    cfd, _ = rc.parse_external(rc.demo_dataset(df))
    cfg = {"keys": ["Coolant_Outlet_Temperature_K", "Q_W"],
           "goals": {"Coolant_Outlet_Temperature_K": ("lower", None), "Q_W": ("higher", None)},
           "weights": {"Coolant_Outlet_Temperature_K": 1, "Q_W": 1}}
    result = rc.analyze(df, codes, {"CFD": cfd}, ["Coolant_Outlet_Temperature_K", "Q_W"], [], [],
                        ["Numerical model", "CFD"], cfg)
    assert set(result["gra"]) == {"Numerical model", "CFD"}
    findings = rc.build_findings(df, result, ["Coolant_Outlet_Temperature_K", "Q_W"])
    assert {f["section"] for f in findings} >= {"model", "compare", "drivers", "gra", "caution"}
    assert ":" not in rc.strip_markup(":blue[**12 W**]")
    blob = rc.export_workbook(df, {"CFD": cfd}, result, findings)
    sheets = pd.ExcelFile(io.BytesIO(blob)).sheet_names
    assert {"Findings", "Numerical model", "Comparison", "Factor contributions", "GRA ranking"} <= set(sheets)
