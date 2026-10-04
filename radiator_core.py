"""Calculation core for the crossflow-radiator DOE / numerical / CFD analysis app.

No input(), print() or Streamlit calls live here, so every function can be
unit-tested and reused from a notebook. The physics, the L50 design and the
ANOVA / GRA maths follow the original notebook script.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Factor, fixed-input and response definitions
# --------------------------------------------------------------------------
FACTOR_SPECS = {
    "water_temperature_K": {
        "label": "Water inlet temperature", "unit": "K",
        "default": [353.15, 358.15, 363.15, 368.15, 373.15],
        "valid": (353.15, 373.15),
        "help": "Hot coolant temperature entering the tubes. Model property tables cover 353.15-373.15 K.",
    },
    "water_velocity_m_s": {
        "label": "Water velocity", "unit": "m/s",
        "default": [0.00625, 0.0109375, 0.015625, 0.0203125, 0.025],
        "help": "Mean coolant speed inside the tubes. Must stay laminar (Re < 2300) for the Nu = 4.36 assumption.",
    },
    "air_temperature_K": {
        "label": "Air inlet temperature", "unit": "K",
        "default": [298.15, 303.15, 308.15, 313.15, 318.15],
        "valid": (298.15, 318.15),
        "help": "Ambient air temperature entering the core. Model property tables cover 298.15-318.15 K.",
    },
    "air_velocity_m_s": {
        "label": "Air face velocity", "unit": "m/s",
        "default": [0.75, 3.5625, 6.375, 9.1875, 12.0],
        "help": "Air speed approaching the core face. It is converted to a maximum in-core velocity for Re and Nu.",
    },
    "fin_width_m": {
        "label": "Fin width (Pw)", "unit": "m",
        "default": [0.016, 0.020, 0.024, 0.028, 0.032],
        "help": "Fin plate dimension along the tube width direction.",
    },
    "fin_thickness_m": {
        "label": "Fin thickness (Pt)", "unit": "m",
        "default": [0.0025, 0.004375, 0.00625, 0.008125, 0.010],
        "help": "Thickness of each fin plate. Fin spacing = Tl/(Np+1) - Pt must stay positive.",
    },
    "tube_width_m": {
        "label": "Tube width (Tw)", "unit": "m",
        "default": [0.016, 0.020, 0.024, 0.028, 0.032],
        "help": "Outer width of the flat oval tube. Must exceed tube height.",
    },
    "tube_height_m": {
        "label": "Tube height (Th)", "unit": "m",
        "default": [0.0025, 0.00275, 0.003, 0.00325, 0.0035],
        "help": "Outer height of the flat oval tube. Must exceed twice the wall thickness.",
    },
    "tube_wall_thickness_m": {
        "label": "Tube wall thickness (Tt)", "unit": "m",
        "default": [0.00025, 0.0002875, 0.000325, 0.0003625, 0.0004],
        "help": "Tube wall thickness. Adds conduction resistance and reduces the water-side flow area.",
    },
}
FACTOR_KEYS = list(FACTOR_SPECS)

FIXED_DEFAULTS = {
    "tube_length_m": ("Tube length Tl", 0.25, "m", float,
                      "Flow length of the tube (also the air-side depth of the core)."),
    "number_of_tubes": ("Number of tubes Nt", 5, "count", int,
                        "Tubes stacked in the fin-height direction."),
    "number_of_plates": ("Number of plates Np", 5, "count", int,
                         "Fin plates along the tube length."),
    "fin_height_m": ("Fin height Ph", 0.25, "m", float,
                     "Fin plate dimension in the tube-stacking direction."),
    "tube_conductivity_W_mK": ("Tube conductivity kt", 230.0, "W/(m K)", float,
                               "Al 1060 H18 assumption."),
    "fin_conductivity_W_mK": ("Fin conductivity kp", 230.0, "W/(m K)", float,
                              "Al 1060 H18 assumption."),
}

RESPONSE_SPECS = {
    "Coolant_Outlet_Temperature_K": {
        "label": "Coolant outlet temperature", "unit": "K", "goal": "lower",
        "help": "Water temperature leaving the radiator. Lower means more heat was removed.",
    },
    "Q_W": {
        "label": "Heat transfer rate Q", "unit": "W", "goal": "higher",
        "help": "Heat rejected from water to air. Higher is better for cooling.",
    },
    "Effectiveness": {
        "label": "Effectiveness", "unit": "-", "goal": "higher",
        "help": "Q divided by the maximum possible heat transfer (0 to 1). Higher means a better-used core.",
    },
}
NUMERICAL_RESPONSE_KEYS = list(RESPONSE_SPECS)
RUN_COUNT = 50


def label_of(key: str) -> str:
    return RESPONSE_SPECS[key]["label"] if key in RESPONSE_SPECS else key.replace("_", " ").strip()


def unit_of(key: str) -> str:
    unit = RESPONSE_SPECS[key]["unit"] if key in RESPONSE_SPECS else ""
    return "" if unit == "-" else unit


def factor_label(key: str) -> str:
    spec = FACTOR_SPECS[key]
    return f"{spec['label']} ({spec['unit']})"


# --------------------------------------------------------------------------
# Taguchi L50 design (column 1 is two-level and unused; columns 2-10 carry
# the nine five-level factors)
# --------------------------------------------------------------------------
L50_TEXT = """
1 1 1 1 1 1 1 1 1 1 1 1
1 1 2 2 2 2 2 2 2 2 2 2
1 1 3 3 3 3 3 3 3 3 3 3
1 1 4 4 4 4 4 4 4 4 4 4
1 1 5 5 5 5 5 5 5 5 5 5
1 2 1 2 3 4 5 1 2 3 4 5
1 2 2 3 4 5 1 2 3 4 5 1
1 2 3 4 5 1 2 3 4 5 1 2
1 2 4 5 1 2 3 4 5 1 2 3
1 2 5 1 2 3 4 5 1 2 3 4
1 3 1 3 5 2 4 4 1 3 5 2
1 3 2 4 1 3 5 5 2 4 1 3
1 3 3 5 2 4 1 1 3 5 2 4
1 3 4 1 3 5 2 2 4 1 3 5
1 3 5 2 4 1 3 3 5 2 4 1
1 4 1 4 2 5 3 5 3 1 4 2
1 4 2 5 3 1 4 1 4 2 5 3
1 4 3 1 4 2 5 2 5 3 1 4
1 4 4 2 5 3 1 3 1 4 2 5
1 4 5 3 1 4 2 4 2 5 3 1
1 5 1 5 4 3 2 4 3 2 1 5
1 5 2 1 5 4 3 5 4 3 2 1
1 5 3 2 1 5 4 1 5 4 3 2
1 5 4 3 2 1 5 2 1 5 4 3
1 5 5 4 3 2 1 3 2 1 5 4
2 1 1 1 4 5 4 3 2 5 2 3
2 1 2 2 5 1 5 4 3 1 3 4
2 1 3 3 1 2 1 5 4 2 4 5
2 1 4 4 2 3 2 1 5 3 5 1
2 1 5 5 3 4 3 2 1 4 1 2
2 2 1 2 1 3 3 2 4 5 5 4
2 2 2 3 2 4 4 3 5 1 1 5
2 2 3 4 3 5 5 4 1 2 2 1
2 2 4 5 4 1 1 5 2 3 3 2
2 2 5 1 5 2 2 1 3 4 4 3
2 3 1 3 3 1 2 5 5 4 2 4
2 3 2 4 4 2 3 1 1 5 3 5
2 3 3 5 5 3 4 2 2 1 4 1
2 3 4 1 1 4 5 3 3 2 5 2
2 3 5 2 2 5 1 4 4 3 1 3
2 4 1 4 5 4 1 2 5 2 3 3
2 4 2 5 1 5 2 3 1 3 4 4
2 4 3 1 2 1 3 4 2 4 5 5
2 4 4 2 3 2 4 5 3 5 1 1
2 4 5 3 4 3 5 1 4 1 2 2
2 5 1 5 2 2 5 3 4 4 3 1
2 5 2 1 3 3 1 4 5 5 4 2
2 5 3 2 4 4 2 5 1 1 5 3
2 5 4 3 5 5 3 1 2 2 1 4
2 5 5 4 1 1 4 2 3 3 2 5
"""
L50 = [tuple(map(int, line.split())) for line in L50_TEXT.strip().splitlines()]


def validate_oa(matrix, columns: int, pair_frequency: int) -> None:
    if not matrix or any(len(row) != columns for row in matrix):
        raise ValueError("DOE array has an invalid shape.")
    for j in range(columns):
        freq = Counter(row[j] for row in matrix)
        if set(freq) != {1, 2, 3, 4, 5} or len(set(freq.values())) != 1:
            raise ValueError(f"DOE column {j + 1} is not balanced across its five levels.")
    for j in range(columns):
        for k in range(j + 1, columns):
            freq = Counter((row[j], row[k]) for row in matrix)
            if len(freq) != 25 or set(freq.values()) != {pair_frequency}:
                raise ValueError(f"DOE columns {j + 1} and {k + 1} are not pairwise orthogonal.")


def build_design() -> np.ndarray:
    """Return the 50 x 9 matrix of zero-based level indices (0..4)."""
    selected = [tuple(row[1:len(FACTOR_KEYS) + 1]) for row in L50]
    validate_oa(selected, len(FACTOR_KEYS), pair_frequency=2)
    return np.array(selected, dtype=int) - 1


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
def validate_levels(levels: dict[str, list[float]]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for the factor-level table."""
    errors, warnings = [], []
    for key, spec in FACTOR_SPECS.items():
        values = levels[key]
        name = spec["label"]
        if len(values) != 5 or not all(v is not None and math.isfinite(v) for v in values):
            errors.append(f"{name}: all five levels must be finite numbers.")
            continue
        if len(set(values)) != 5:
            errors.append(f"{name}: the five levels must be distinct.")
        elif list(values) != sorted(values):
            errors.append(f"{name}: enter levels in increasing order.")
        if min(values) <= 0:
            errors.append(f"{name}: levels must be positive.")
        if "valid" in spec:
            lo, hi = spec["valid"]
            if min(values) < lo or max(values) > hi:
                errors.append(f"{name}: property tables only cover {lo:g}-{hi:g} {spec['unit']}.")
    return errors, warnings


def validate_fixed(fixed: dict[str, float | int]) -> list[str]:
    errors = []
    for key, (label, _, _, _, _) in FIXED_DEFAULTS.items():
        value = fixed.get(key)
        if value is None or not math.isfinite(value) or value <= 0:
            errors.append(f"{label} must be a positive finite value.")
    return errors


def check_doe_geometry(cases: list[dict[str, float]], fixed: dict) -> list[str]:
    """Flag geometrically impossible runs before any data are collected."""
    Tl, Nt = float(fixed["tube_length_m"]), int(fixed["number_of_tubes"])
    Np, Ph = int(fixed["number_of_plates"]), float(fixed["fin_height_m"])
    invalid = []
    for run, case in enumerate(cases, start=1):
        Tw, Th, Tt, Pt = (case["tube_width_m"], case["tube_height_m"],
                          case["tube_wall_thickness_m"], case["fin_thickness_m"])
        reasons = []
        if not (Tw > Th > 2 * Tt > 0):
            reasons.append("require tube width > tube height > 2 x wall thickness")
        if Ph / (Nt + 1) - Th <= 0:
            reasons.append("tube spacing is non-positive")
        if Tl / (Np + 1) - Pt <= 0:
            reasons.append("fin spacing is non-positive")
        if reasons:
            invalid.append(f"Run {run}: " + "; ".join(reasons))
    return invalid


# --------------------------------------------------------------------------
# Numerical model (unchanged physics from the notebook)
# --------------------------------------------------------------------------
def piecewise_linear(x: float, points) -> float:
    if x <= points[0][0]:
        (x0, y0), (x1, y1) = points[0], points[1]
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    (x0, y0), (x1, y1) = points[-2], points[-1]
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def calculate_one(case: dict[str, float], fixed: dict) -> dict[str, float | bool]:
    Vw, Twi = case["water_velocity_m_s"], case["water_temperature_K"]
    Va, Tai = case["air_velocity_m_s"], case["air_temperature_K"]
    Pw, Pt = case["fin_width_m"], case["fin_thickness_m"]
    Tw, Th, Tt = case["tube_width_m"], case["tube_height_m"], case["tube_wall_thickness_m"]
    Tl, Nt = float(fixed["tube_length_m"]), int(fixed["number_of_tubes"])
    Np, Ph = int(fixed["number_of_plates"]), float(fixed["fin_height_m"])
    kt, kp = float(fixed["tube_conductivity_W_mK"]), float(fixed["fin_conductivity_W_mK"])

    if not (Tw > Th > 2 * Tt > 0):
        raise ValueError("tube geometry invalid: require Tw > Th > 2*Tt > 0")
    if min(Vw, Va, Twi, Tai, Pw, Pt, Tl, Ph, kt, kp) <= 0:
        raise ValueError("temperatures, velocities, dimensions and conductivities must be positive")
    if not 353.15 <= Twi <= 373.15:
        raise ValueError("water property interpolation is supported only from 353.15 K to 373.15 K")
    if not 298.15 <= Tai <= 318.15:
        raise ValueError("air property interpolation is supported only from 298.15 K to 318.15 K")

    Cpw = piecewise_linear(Twi, ((353.15, 4197.0), (363.15, 4205.0), (373.15, 4216.0)))
    rho_w = piecewise_linear(Twi, ((353.15, 971.8), (363.15, 965.3), (373.15, 958.4)))
    kw = piecewise_linear(Twi, ((353.15, 0.670), (363.15, 0.675), (373.15, 0.679)))
    mu_w = piecewise_linear(Twi, ((353.15, 3.55e-4), (363.15, 3.15e-4), (373.15, 2.82e-4)))
    Prw = mu_w * Cpw / kw

    Cpa = piecewise_linear(Tai, ((298.15, 1006.5), (308.15, 1006.7), (318.15, 1007.0)))
    rho_a = 101325.0 / (287.05 * Tai)
    ka = piecewise_linear(Tai, ((298.15, 0.0262), (308.15, 0.0269), (318.15, 0.0276)))
    mu_a = piecewise_linear(Tai, ((298.15, 1.837e-5), (308.15, 1.884e-5), (318.15, 1.931e-5)))
    Pra = mu_a * Cpa / ka

    Opt = math.pi * Th + 2 * (Tw - Th)
    Wpt = math.pi * (Th - 2 * Tt) + 2 * ((Tw - 2 * Tt) - (Th - 2 * Tt))
    At = Nt * Opt * Tl
    Ati = Nt * Wpt * Tl
    Act = (Tw - 2 * Tt - Th) * (Th - 2 * Tt) + math.pi * ((Th - Tt) / 2) ** 2
    Acto = (Tw - Th) * Th + math.pi * ((Th - Tt) / 2) ** 2
    Ato = At - Np * Nt * 2 * Acto
    Ap = Np * 2 * ((Pw * Ph) + (Pt * Ph) + (Pw * Pt)) - Np * Nt * 2 * Acto
    Ar = Ato + Ap

    St = Ph / (Nt + 1) - Th
    Sp = Tl / (Np + 1) - Pt
    edge_count = ((Nt + 1) * (Np + 1) - (Nt - 1) * (Np - 1)) / 2
    Apa = ((Nt - 1) * (Np - 1) * St * Sp
           + edge_count * (St + Th / 2) * (Sp + Pt / 2))
    Awpa = (2 * (Nt - 1) * (Np - 1) * St
            + 2 * (Nt - 1) * (Np - 1) * Sp
            + 2 * edge_count * (St + Th / 2)
            + 2 * edge_count * (Sp + Pt / 2))
    if min(Act, Ati, Ato, Ap, Ar, St, Sp, Apa, Awpa) <= 0:
        raise ValueError("geometry equations produced a non-positive area/spacing/perimeter")

    Dht = 4 * Act / Wpt
    Dha = 4 * Apa / Awpa
    Dexp = math.sqrt(Tl * mu_w / (0.05 * rho_w * Vw * Prw))
    Rew = rho_w * Vw * Dht / mu_w
    Vmax_air = Va * Tl * Ph / Apa
    Rea = rho_a * Vmax_air * Dha / mu_a
    developed = Dht < Dexp
    if Rew >= 2300 or not (0.5 < Prw < 100) or not developed:
        raise ValueError(f"water Nu=4.36 criteria fail: Rew={Rew:.3g}, Prw={Prw:.3g}, developed={developed}")
    if Rea >= 3.5e5:
        raise ValueError(f"air Reynolds number {Rea:.3g} is outside configured range")

    Nuw = 4.36
    Nua = 0.644 * math.sqrt(Rea) * Pra ** (1 / 3)
    hw = Nuw * kw / Dht
    ha = Nua * ka / Dha
    m = math.sqrt(2 * ha / (kp * Pt))
    Lc = St / 2 + Pt / 2
    nf = math.tanh(m * Lc) / (m * Lc)
    n0 = 1 - (Ap / Ar) * (1 - nf)
    UA = 1 / (1 / (n0 * ha * Ar) + Tt / (kt * ((At + Ati) / 2)) + 1 / (hw * Ati))
    Cw = rho_w * Vw * Act * Nt * Cpw
    Ca = rho_a * Va * Tl * Ph * Cpa
    Cmin, Cmax = min(Cw, Ca), max(Cw, Ca)
    Cr = Cmin / Cmax
    NTU = UA / Cmin
    eps = 1 - math.exp((NTU ** 0.22 / Cr) * (math.exp(-Cr * NTU ** 0.78) - 1))
    Q = eps * Cmin * (Twi - Tai)
    Tco = Twi - Q / Cw
    Tao = Tai + Q / Ca
    return {
        "Rew": Rew, "Rea": Rea, "Dht_m": Dht, "Dexp_m": Dexp,
        "UA_W_K": UA, "Cw_W_K": Cw, "Ca_W_K": Ca, "Cr": Cr, "NTU": NTU,
        "Coolant_Outlet_Temperature_K": Tco, "Q_W": Q, "Effectiveness": eps,
        "Air_Outlet_Temperature_K": Tao,
        "energy_balance_error_W": Cw * (Twi - Tco) - Ca * (Tao - Tai),
    }


def run_model(levels: dict[str, list[float]], fixed: dict):
    """Build the DOE and evaluate the model for all 50 runs.

    Returns (model_df, codes, invalid_geometry, failures). model_df is None
    when any run is invalid or cannot be calculated, because the balanced
    contribution analysis needs all 50 runs.
    """
    codes = build_design()
    cases = [{k: levels[k][codes[i, j]] for j, k in enumerate(FACTOR_KEYS)}
             for i in range(len(codes))]
    invalid = check_doe_geometry(cases, fixed)
    if invalid:
        return None, codes, invalid, []
    records, failures = [], []
    for i, case in enumerate(cases):
        try:
            records.append({"DOE_Run": i + 1, **case, **calculate_one(case, fixed)})
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            failures.append(f"Run {i + 1}: {exc}")
    if failures:
        return None, codes, [], failures
    return pd.DataFrame(records), codes, [], []


# --------------------------------------------------------------------------
# External (CFD / experiment) data parsing
# --------------------------------------------------------------------------
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_ALIASES = {
    "Coolant_Outlet_Temperature_K": {
        "coolantoutlettemperaturek", "coolantoutlettemperature", "tco", "tcok",
        "wateroutlettemperature", "wateroutlettemperaturek", "coolantouttemp",
        "coolantoutlettempk", "toutlet", "toutletk"},
    "Q_W": {"qw", "q", "heattransferrate", "heattransferratew", "heatduty",
            "heatrejection", "heatrejectionw"},
    "Effectiveness": {"effectiveness", "eps", "epsilon"},
}
_RUN_NAMES = {"run", "doerun", "runno", "runnumber", "runid"}


def parse_external(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate an uploaded/pasted table and return (clean_df, notes).

    Recognised columns: a run number (optional), Coolant_Outlet_Temperature_K,
    Q_W, Effectiveness. Any other fully numeric column becomes an extra
    response (for example air pressure drop). DOE factor columns are ignored.
    """
    notes: list[str] = []
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    if len(df) != RUN_COUNT:
        raise ValueError(f"Expected {RUN_COUNT} rows (one per DOE run); found {len(df)}.")

    run_col = next((c for c in df.columns if _norm(c) in _RUN_NAMES), None)
    if run_col is not None:
        runs = pd.to_numeric(df[run_col], errors="coerce")
        if runs.isna().any() or sorted(runs.astype(int)) != list(range(1, RUN_COUNT + 1)):
            raise ValueError(f"The Run column must contain each of 1 to {RUN_COUNT} exactly once.")
        df = df.assign(_run=runs.astype(int)).sort_values("_run").reset_index(drop=True)
        run_values = df.pop("_run").to_numpy()
    else:
        notes.append("No Run column found; rows were taken as DOE Run 1 to 50 in the order given.")
        df = df.reset_index(drop=True)
        run_values = np.arange(1, RUN_COUNT + 1)

    factor_norms = {_norm(k) for k in FACTOR_KEYS}
    out = pd.DataFrame({"DOE_Run": run_values})
    used = {run_col} if run_col is not None else set()
    empty_cols: list[str] = []
    for key, aliases in _ALIASES.items():
        col = next((c for c in df.columns
                    if c not in used and (_norm(c) in aliases or _norm(c) == _norm(key))), None)
        if col is None:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().all():
            notes.append(f"Column '{col}' is empty, so {label_of(key)} was skipped.")
            used.add(col)
            empty_cols.append(col)
            continue
        bad = np.where(~np.isfinite(values.to_numpy(dtype=float)))[0] + 1
        if len(bad):
            raise ValueError(f"'{col}' has missing or non-numeric values in row(s) "
                             f"{', '.join(map(str, bad[:8]))}{'...' if len(bad) > 8 else ''}.")
        arr = values.to_numpy(dtype=float)
        if key == "Coolant_Outlet_Temperature_K" and arr.max() < 200:
            raise ValueError(f"'{col}' looks like degrees Celsius (largest value {arr.max():.1f}). "
                             "Convert to kelvin (add 273.15) so it matches the model.")
        if key == "Effectiveness" and (arr.min() < 0 or arr.max() > 1):
            notes.append(f"'{col}' has values outside 0 to 1, which an effectiveness cannot have. "
                         "Check the column.")
        out[key] = arr
        used.add(col)

    for col in df.columns:
        if col in used or _norm(col) in factor_norms or not col or col.startswith("Unnamed"):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().all() and np.isfinite(values.to_numpy(dtype=float)).all():
            out[col] = values.to_numpy(dtype=float)
            notes.append(f"Extra column '{col}' was kept as an additional response.")
        else:
            notes.append(f"Column '{col}' was ignored (not fully numeric).")

    if not any(k in out.columns for k in NUMERICAL_RESPONSE_KEYS):
        if empty_cols:
            raise ValueError("The response columns are still empty. Fill in the CFD/test results "
                             "for each run, then upload the file again.")
        raise ValueError("No recognised response column found. Use Coolant_Outlet_Temperature_K "
                         "and/or Q_W (or Effectiveness).")
    return out, notes


def run_sheet(model_df: pd.DataFrame) -> pd.DataFrame:
    """Blank result sheet: DOE settings per run plus empty response columns."""
    sheet = model_df[["DOE_Run"] + FACTOR_KEYS].rename(columns={"DOE_Run": "Run"})
    sheet["Coolant_Outlet_Temperature_K"] = np.nan
    sheet["Q_W"] = np.nan
    return sheet


def soft_geometry_warnings(model_df: pd.DataFrame) -> list[str]:
    """Checks the model does not enforce but a designer should see."""
    out = []
    narrow = model_df.loc[model_df["fin_width_m"] < model_df["tube_width_m"], "DOE_Run"]
    if len(narrow):
        out.append(
            f"In {len(narrow)} of {len(model_df)} runs the fin is narrower than the tube "
            f"(fin width < tube width), so the tube would stick out past the fin. The model still "
            f"calculates these runs because it does not check this. Runs: "
            f"{', '.join(map(str, narrow.tolist()))}."
        )
    if (model_df["Cw_W_K"] <= model_df["Ca_W_K"]).all():
        out.append("The water side limits heat transfer (it has the smaller heat-capacity rate) in "
                   "every run. Air-side factors will matter less than they would in a design where "
                   "air is the limiting side.")
    return out


def default_goal(key: str) -> str:
    """Suggested GRA direction for an output."""
    if key in RESPONSE_SPECS:
        return RESPONSE_SPECS[key]["goal"]
    low = key.lower()
    if any(t in low for t in ("pressure", "drop", "loss", "power", "pump", "fan", "mass", "cost")):
        return "lower"
    return "higher"


def pretty_name(key: str) -> str:
    """Readable name for a model response or an extra uploaded column."""
    return label_of(key)


def demo_dataset(model_df: pd.DataFrame, kind: str = "CFD", seed: int = 7) -> pd.DataFrame:
    """Synthetic stand-in for CFD/test results so the app can be tried without data.

    Heat transfer is the model value scaled and perturbed; outlet temperature and
    effectiveness are derived from it with the model's own heat-capacity rates, so
    the three outputs stay energy-consistent. Not a measurement.
    """
    rng = np.random.default_rng(seed + (0 if kind == "CFD" else 11))
    scale, noise = (0.93, 0.04) if kind == "CFD" else (1.04, 0.08)
    q_model = model_df["Q_W"].to_numpy()
    q = q_model * scale + rng.normal(0, noise * (q_model.max() - q_model.min()), len(q_model))
    q = np.clip(q, 0.02 * q_model.max(), None)
    cw = model_df["Cw_W_K"].to_numpy()
    cmin = np.minimum(cw, model_df["Ca_W_K"].to_numpy())
    dt = (model_df["water_temperature_K"] - model_df["air_temperature_K"]).to_numpy()
    rea = model_df["Rea"].to_numpy()
    dp = 0.0045 * rea ** 1.15 * (1 + rng.normal(0, 0.05, len(rea)))
    return pd.DataFrame({
        "Run": model_df["DOE_Run"].to_numpy(),
        "Coolant_Outlet_Temperature_K": model_df["water_temperature_K"].to_numpy() - q / cw,
        "Q_W": q,
        "Effectiveness": np.clip(q / (cmin * dt), 0, 1),
        "Air_pressure_drop_Pa": dp,
    })


def strip_markup(text: str) -> str:
    """Remove Streamlit colour markup (:blue[**x**]) for plain-text exports."""
    text = re.sub(r":[a-z-]+\[(.*?)\]", r"\1", text)
    return text.replace("**", "")


def export_workbook(model_df: pd.DataFrame, datasets: dict, analysis: dict, findings: list[dict]) -> bytes:
    """Build one Excel workbook with the model table, external data and every result table."""
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"Finding": [f["title"] for f in findings],
                      "Detail": [strip_markup(f["text"]) for f in findings]}).to_excel(
            xw, sheet_name="Findings", index=False)
        model_df.to_excel(xw, sheet_name="Numerical model", index=False)
        for name, ext in datasets.items():
            ext.to_excel(xw, sheet_name=re.sub(r"[\[\]:*?/\\]", "_", f"Data {name}")[:31], index=False)
        rows = []
        for name, per in analysis["comparison"].items():
            for key, m in per.items():
                rows.append({"Dataset": name, "Output": pretty_name(key), "Unit": unit_of(key),
                             "Mean error (ext - model)": m["bias"], "MAE": m["mae"], "RMSE": m["rmse"],
                             "Max abs error": m["max_abs_error"], "NRMSE (%)": m["nrmse_pct"],
                             "MAPE (%)": m["mape_pct"], "Correlation r": m["r"],
                             "Agreement": agreement_rating(m["nrmse_pct"])})
        if rows:
            pd.DataFrame(rows).to_excel(xw, sheet_name="Comparison", index=False)
        contrib = []
        for name, per in analysis["anova"].items():
            for key, (table, meta) in per.items():
                t = table.drop(columns=["key"]).copy()
                t.insert(0, "Output", pretty_name(key))
                t.insert(0, "Dataset", name)
                t["Main-effects R2 (%)"] = meta["r2_pct"]
                contrib.append(t)
        if contrib:
            pd.concat(contrib).to_excel(xw, sheet_name="Factor contributions", index=False)
        gra_rows = []
        for name, g in analysis["gra"].items():
            t = g["ranking"].copy()
            t.insert(0, "Dataset", name)
            gra_rows.append(t)
        if gra_rows:
            pd.concat(gra_rows).to_excel(xw, sheet_name="GRA ranking", index=False)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Comparison metrics
# --------------------------------------------------------------------------
def compare_metrics(model_values, external_values) -> dict[str, float | None]:
    """Paired metrics; errors are external minus numerical."""
    m = np.asarray(model_values, float)
    e = np.asarray(external_values, float)
    err = e - m
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ext_range = float(e.max() - e.min())
    r = None
    if m.std() > 0 and e.std() > 0:
        r = float(np.corrcoef(m, e)[0, 1])
    nonzero = e != 0
    mape = float(100 * np.mean(np.abs(err[nonzero] / e[nonzero]))) if nonzero.any() else None
    return {
        "bias": float(err.mean()), "mae": float(np.abs(err).mean()), "rmse": rmse,
        "max_abs_error": float(np.abs(err).max()),
        "nrmse_pct": None if ext_range == 0 else 100 * rmse / ext_range,
        "mape_pct": mape, "r": r,
        "bias_pct_of_model_mean": float(100 * err.mean() / m.mean()) if m.mean() != 0 else None,
    }


def agreement_rating(nrmse_pct: float | None) -> str:
    """Rule-of-thumb band for NRMSE (% of the external data range)."""
    if nrmse_pct is None:
        return "n/a"
    if nrmse_pct < 5:
        return "Good"
    if nrmse_pct < 15:
        return "Moderate"
    return "Poor"


# --------------------------------------------------------------------------
# Main-effects ANOVA (balanced L50: 10 runs per level, 4 df per factor)
# --------------------------------------------------------------------------
def main_effects_anova(codes: np.ndarray, y, factor_keys=FACTOR_KEYS):
    y = np.asarray(y, float)
    n = len(y)
    if n != RUN_COUNT:
        raise ValueError(f"ANOVA requires all {RUN_COUNT} DOE runs; received {n}.")
    grand = y.mean()
    ss_total = float(((y - grand) ** 2).sum())
    if ss_total <= 0:
        raise ValueError("The response has no variation; contribution percentages are undefined.")
    rows, level_means, ss_sum = [], {}, 0.0
    for j, key in enumerate(factor_keys):
        means, ss = [], 0.0
        for lvl in range(5):
            sel = codes[:, j] == lvl
            if sel.sum() != 10:
                raise ValueError(f"{key} level {lvl + 1} occurs {int(sel.sum())} times; expected 10.")
            mean = y[sel].mean()
            means.append(float(mean))
            ss += 10 * (mean - grand) ** 2
        level_means[key] = means
        ss_sum += ss
        rows.append({"key": key, "Factor": factor_label(key), "SS": float(ss), "df": 4,
                     "MS": float(ss / 4), "Range of level means": max(means) - min(means)})
    df_resid = n - 1 - 4 * len(factor_keys)
    if df_resid <= 0:
        raise ValueError("Not enough residual degrees of freedom.")
    ss_res = max(0.0, ss_total - ss_sum)
    ms_res = ss_res / df_resid
    for row in rows:
        row["Contribution (%)"] = 100 * row["SS"] / ss_total
        row["F (screening)"] = row["MS"] / ms_res if ms_res > 0 else np.nan
    table = (pd.DataFrame(rows).sort_values("Contribution (%)", ascending=False)
             .reset_index(drop=True))
    table.insert(0, "Rank", np.arange(1, len(table) + 1))
    residual_pct = 100 * ss_res / ss_total
    meta = {"ss_total": ss_total, "df_resid": df_resid, "residual_pct": residual_pct,
            "r2_pct": 100 - residual_pct, "level_means": level_means}
    return table, meta


# --------------------------------------------------------------------------
# Grey relational analysis
# --------------------------------------------------------------------------
def normalize_for_gra(values, goal: tuple[str, float | None]) -> np.ndarray:
    mode, target = goal
    v = np.asarray(values, float)
    if mode == "target":
        if target is None:
            raise ValueError("A target value is required for target-based GRA normalization.")
        dev = np.abs(v - target)
        return np.ones_like(v) if dev.max() == 0 else 1 - dev / dev.max()
    lo, hi = v.min(), v.max()
    if hi == lo:
        return np.ones_like(v)
    return (hi - v) / (hi - lo) if mode == "lower" else (v - lo) / (hi - lo)


def calculate_gra(df: pd.DataFrame, keys, goals, weights, zeta: float = 0.5):
    norm = {k: normalize_for_gra(df[k], goals[k]) for k in keys}
    dev = {k: np.abs(1 - norm[k]) for k in keys}
    all_dev = np.concatenate([dev[k] for k in keys])
    dmin, dmax = all_dev.min(), all_dev.max()
    if dmax == 0:
        coef = {k: np.ones(len(df)) for k in keys}
    else:
        coef = {k: (dmin + zeta * dmax) / (dev[k] + zeta * dmax) for k in keys}
    grades = sum(weights[k] * coef[k] for k in keys)
    return np.asarray(grades), coef


def subset_gra(cfg: dict, available: list[str]) -> dict | None:
    keys = [k for k in cfg["keys"] if k in available]
    if len(keys) < 2:
        return None
    w = {k: float(cfg["weights"][k]) for k in keys}
    total = sum(w.values())
    return {"keys": keys, "goals": {k: cfg["goals"][k] for k in keys},
            "weights": {k: v / total for k, v in w.items()}}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def analyze(model_df, codes, datasets: dict[str, pd.DataFrame], comparison_keys: list[str],
            model_only_keys: list[str], extra_keys: list[str], detailed: list[str],
            gra_cfg: dict | None) -> dict:
    """Run comparison, contribution and GRA analyses.

    `detailed` lists dataset names (plus "Numerical model") for the factor
    contribution and GRA steps. Everything is returned in one dict.
    """
    result = {"comparison": {}, "anova": {}, "gra": {}, "notes": []}
    for name, ext in datasets.items():
        result["comparison"][name] = {
            k: compare_metrics(model_df[k], ext[k]) for k in comparison_keys if k in ext.columns
        }

    sets = {}
    for name in detailed:
        if name == "Numerical model":
            sets[name] = (model_df, comparison_keys + [k for k in model_only_keys
                                                       if k not in comparison_keys])
        else:
            ext = datasets[name]
            sets[name] = (ext, [k for k in comparison_keys + extra_keys if k in ext.columns])

    for name, (frame, keys) in sets.items():
        result["anova"][name] = {}
        for key in keys:
            try:
                result["anova"][name][key] = main_effects_anova(codes, frame[key].to_numpy())
            except ValueError as exc:
                result["notes"].append(f"Contribution analysis skipped for {label_of(key)} in {name}: {exc}")
        if gra_cfg:
            cfg = subset_gra(gra_cfg, keys)
            if cfg is None:
                have = ", ".join(label_of(k) for k in keys) or "none"
                result["notes"].append(
                    f"GRA was skipped for {name}: it holds only {len(keys)} of the selected outputs ({have}) "
                    "and GRA needs at least two.")
                continue
            try:
                grades, coef = calculate_gra(frame, cfg["keys"], cfg["goals"], cfg["weights"])
                table, meta = main_effects_anova(codes, grades)
                best = int(np.argmax(grades))
                result["gra"][name] = {
                    "cfg": cfg, "grades": grades, "coef": coef, "table": table, "meta": meta,
                    "best_index": best,
                    "ranking": pd.DataFrame({"DOE_Run": frame["DOE_Run"], "GRA grade": grades})
                               .sort_values("GRA grade", ascending=False).reset_index(drop=True),
                }
            except ValueError as exc:
                result["notes"].append(f"GRA skipped for {name}: {exc}")
    return result


# --------------------------------------------------------------------------
# Plain-language findings
# --------------------------------------------------------------------------
def build_findings(model_df: pd.DataFrame, analysis: dict, comparison_keys: list[str]) -> list[dict]:
    """Return a list of {kind, title, text}; kind is success / info / warning."""
    out: list[dict] = []
    q, tco, eps = model_df["Q_W"], model_df["Coolant_Outlet_Temperature_K"], model_df["Effectiveness"]
    water_limited = int((model_df["Cw_W_K"] <= model_df["Ca_W_K"]).sum())
    out.append({
        "kind": "info", "section": "model", "title": "What the numerical model predicts",
        "text": (f"Across the 50 DOE runs heat transfer ranges from :blue[**{q.min():.0f} to {q.max():.0f} W**], "
                 f"coolant outlet temperature from :blue[**{tco.min():.1f} to {tco.max():.1f} K**] and "
                 f"effectiveness from :blue[**{eps.min():.3f} to {eps.max():.3f}**]. "
                 f"The water side has the smaller heat-capacity rate (Cmin) in {water_limited} of 50 runs, "
                 f"the air side in {50 - water_limited}."),
    })

    for name, per_key in analysis["comparison"].items():
        for key, m in per_key.items():
            rating = agreement_rating(m["nrmse_pct"])
            kind = {"Good": "success", "Moderate": "info", "Poor": "warning"}.get(rating, "info")
            unit = unit_of(key)
            direction = "higher" if m["bias"] > 0 else "lower"
            pct = m["bias_pct_of_model_mean"]
            pct_txt = f" ({pct:+.1f}% of the model mean)" if pct is not None else ""
            nr = "n/a" if m["nrmse_pct"] is None else f"{m['nrmse_pct']:.1f}%"
            bias_txt = f"{abs(m['bias']):.4g}" + (f" {unit}" if unit else "")
            text = (f"{name} reads on average :orange[**{bias_txt} {direction}**] than the numerical "
                    f"model{pct_txt}. NRMSE is :blue[**{nr}**] of the {name} range, which is a "
                    f":{ {'Good': 'green', 'Moderate': 'orange', 'Poor': 'red'}.get(rating, 'gray') }[**{rating.lower()}**] match "
                    f"by the usual 5% / 15% rule of thumb.")
            if m["r"] is not None:
                if m["r"] >= 0.95 and rating != "Good":
                    text += (f" The correlation is high (r = {m['r']:.3f}), so the two sources follow the same trend "
                             "and the gap is mostly a systematic offset.")
                elif m["r"] < 0.8:
                    text += (f" The correlation is weak (r = {m['r']:.3f}), so the sources disagree about how the "
                             "response changes between runs, not only about its level.")
            out.append({"kind": kind, "section": "compare", "title": f"{label_of(key)}: model vs {name}", "text": text})

    keys_seen = list(dict.fromkeys(k for per in analysis["anova"].values() for k in per))
    for key in keys_seen:
        leaders = {}
        for name, per in analysis["anova"].items():
            if key in per:
                table, meta = per[key]
                top = table.iloc[0]
                second = table.iloc[1]
                leaders[name] = top["Factor"]
                text = (f"In {name}, :violet[**{top['Factor']}**] explains :blue[**{top['Contribution (%)']:.1f}%**] of the "
                        f"variation, followed by {second['Factor']} ({second['Contribution (%)']:.1f}%). "
                        f"Main effects account for {meta['r2_pct']:.1f}% in total.")
                if meta["residual_pct"] > 10:
                    text += (f" The remaining :orange[{meta['residual_pct']:.1f}%] is interaction or non-additive "
                             "behaviour, so treat the ranking as a guide and not a full explanation.")
                out.append({"kind": "info", "section": "drivers", "title": f"Main drivers of {label_of(key)} ({name})", "text": text})
        if len(set(leaders.values())) > 1:
            parts = "; ".join(f"{n}: {f}" for n, f in leaders.items())
            out.append({"kind": "warning", "section": "drivers", "title": f"Datasets disagree on the leading factor for {label_of(key)}",
                        "text": f"{parts}. Check whether the CFD/test setup captures the same physics as the model."})

    for name, g in analysis["gra"].items():
        run = int(g["ranking"].iloc[0]["DOE_Run"])
        row = model_df.loc[model_df["DOE_Run"] == run].iloc[0]
        lead = g["table"].iloc[0]
        goals = ", ".join(f"{label_of(k)} {'target ' + format(v[1], 'g') if v[0] == 'target' else v[0]}"
                          for k, v in g["cfg"]["goals"].items())
        setting = "; ".join(f"{FACTOR_SPECS[k]['label']} = {row[k]:g} {FACTOR_SPECS[k]['unit']}"
                            for k in FACTOR_KEYS)
        out.append({
            "kind": "success", "section": "gra", "title": f"Best DOE run by combined GRA grade ({name})",
            "text": (f":green[**Run {run}**] scores highest (grade {g['grades'][g['best_index']]:.3f}) for the goals "
                     f"{goals}. The factor that moves the grade most is :violet[**{lead['Factor']}**] "
                     f"({lead['Contribution (%)']:.1f}%). Settings of that run: {setting}."),
        })

    for note in analysis["notes"]:
        out.append({"kind": "warning", "section": "caution", "title": "Skipped step", "text": note})
    out.append({
        "kind": "info", "section": "caution", "title": "How to read these results",
        "text": ("Contribution percentages describe main effects over the five levels you entered, not a physical law. "
                 "The best GRA run is the best of the 50 sampled settings, not a continuous optimum. F values are "
                 "screening aids because the model is deterministic. Replicated experiments need a separate "
                 "pure-error analysis."),
    })
    return out
