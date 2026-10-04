"""Crossflow Radiator DOE Analyzer (Streamlit).

Compares a correlation-based numerical model with CFD or experimental results over
a 50-run Taguchi L50 design, ranks the inputs that drive each output, and scores
runs with grey relational analysis (GRA).

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import radiator_core as rc

MODEL = "Numerical model"
DEFAULT_NAMES = ["CFD", "Experiment", "Dataset 3"]
MAX_EXTERNAL = 3

st.set_page_config(
    page_title="Radiator DOE Analyzer",
    page_icon=":material/thermostat:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 4.2rem; max-width: 1360px;}
[data-testid="stMetricValue"] {font-size: 1.5rem; line-height: 1.25;}
[data-testid="stMetricValue"] > div {white-space: normal; overflow: visible; text-overflow: clip;}
.rda-hero h1 {font-size: 2rem; margin: 0; padding: 0; letter-spacing: -0.01em;}
.rda-hero p {margin: .35rem 0 0; opacity: .78; font-size: 1.02rem; max-width: 60rem;}
.rda-steps {display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; margin: 1rem 0 1.1rem;}
.rda-step {border: 1px solid rgba(128,128,128,.30); border-radius: .6rem; padding: .6rem .8rem;
           font-size: .92rem; line-height: 1.35;}
.rda-step b {display: block; font-size: .74rem; text-transform: uppercase; letter-spacing: .07em;
             opacity: .62; margin-bottom: .15rem;}
@media (max-width: 800px) {.rda-steps {grid-template-columns: 1fr 1fr;}}
</style>
<div class="rda-hero">
  <h1>Radiator DOE Analyzer</h1>
  <p>Compare a numerical model with CFD or test results over a 50-run design, see which inputs
  drive each output, and find the best run.</p>
</div>
<div class="rda-steps">
  <div class="rda-step"><b>Step 1 &middot; Inputs</b>Set five levels for each of nine inputs.</div>
  <div class="rda-step"><b>Step 2 &middot; Data</b>Add CFD or test results for the 50 runs (optional).</div>
  <div class="rda-step"><b>Step 3 &middot; Choose</b>Pick outputs, datasets and GRA goals in the sidebar.</div>
  <div class="rda-step"><b>Step 4 &middot; Results</b>Read the summary, comparison, drivers and best run.</div>
</div>
""",
    unsafe_allow_html=True,
)

if "levels_ver" not in st.session_state:
    st.session_state.levels_ver = 0


# ---------------------------------------------------------------------------
# Chart styling (palette from the data-viz reference: fixed categorical order,
# validated for 4 series on both surfaces)
# ---------------------------------------------------------------------------
def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def ink() -> dict:
    if theme_mode() == "dark":
        return dict(bg="#0e1117", text="#c3c2b7", muted="#898781", grid="#2c2c2a", base="#383835",
                    gray="#52514e", series=["#3987e5", "#d95926", "#199e70", "#c98500"])
    return dict(bg="#ffffff", text="#52514e", muted="#898781", grid="#e1e0d9", base="#c3c2b7",
                gray="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])


def series_color(name: str, order: list[str]) -> str:
    c = ink()["series"]
    return c[order.index(name) % len(c)] if name in order else c[0]


def style(fig: go.Figure, height: int, legend: bool = True) -> go.Figure:
    c = ink()
    fig.update_layout(
        height=height, margin=dict(l=8, r=12, t=34 if legend else 12, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=13, color=c["text"]),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(gridcolor=c["grid"], gridwidth=1, zeroline=False, linecolor=c["base"],
                     tickfont=dict(color=c["muted"]), automargin=True)
    fig.update_yaxes(gridcolor=c["grid"], gridwidth=1, zeroline=False, linecolor=c["base"],
                     tickfont=dict(color=c["muted"]), automargin=True)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def out_label(key: str) -> str:
    unit = rc.unit_of(key)
    return f"{rc.label_of(key)} ({unit})" if unit and unit != "-" else rc.label_of(key)


def num(x, digits: int = 4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{digits}g}"


def parity_fig(model, ext, runs, key: str, name: str, color: str) -> go.Figure:
    c = ink()
    lo, hi = float(min(model.min(), ext.min())), float(max(model.max(), ext.max()))
    pad = 0.05 * ((hi - lo) or 1.0)
    rng = [lo - pad, hi + pad]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rng, y=rng, mode="lines", name="Perfect agreement", hoverinfo="skip",
                             line=dict(color=c["muted"], width=1.5)))
    fig.add_trace(go.Scatter(
        x=model, y=ext, mode="markers", name=name, customdata=runs,
        marker=dict(size=9, color=color, line=dict(width=2, color=c["bg"])),
        hovertemplate=f"Run %{{customdata}}<br>Model %{{x:.4g}}<br>{name} %{{y:.4g}}<extra></extra>"))
    unit = rc.unit_of(key)
    suffix = f" ({unit})" if unit else ""
    fig.update_xaxes(title=f"Model{suffix}", range=rng)
    fig.update_yaxes(title=f"{name}{suffix}", range=rng)
    return style(fig, 420)


def runs_fig(series: dict[str, np.ndarray], runs, key: str, order: list[str]) -> go.Figure:
    c = ink()
    fig = go.Figure()
    for name, vals in series.items():
        fig.add_trace(go.Scatter(
            x=runs, y=vals, mode="markers", name=name,
            marker=dict(size=8, color=series_color(name, order), line=dict(width=2, color=c["bg"])),
            hovertemplate=f"Run %{{x}}<br>{name}: %{{y:.4g}}<extra></extra>"))
    fig.update_xaxes(title="DOE run", dtick=5)
    fig.update_yaxes(title=out_label(key))
    return style(fig, 360)


def error_fig(errors: dict[str, np.ndarray], runs, key: str, order: list[str]) -> go.Figure:
    c = ink()
    names = list(errors)
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=[f"{n} minus numerical model" for n in names])
    for i, name in enumerate(names, start=1):
        fig.add_trace(go.Bar(
            x=runs, y=errors[name], name=name, showlegend=False,
            marker=dict(color=series_color(name, order), line=dict(width=1, color=c["bg"])),
            hovertemplate=f"Run %{{x}}<br>Error %{{y:.4g}} {rc.unit_of(key)}<extra></extra>"), row=i, col=1)
        fig.update_yaxes(title=rc.unit_of(key) or None, row=i, col=1, zeroline=True, zerolinecolor=c["base"])
    fig.update_xaxes(dtick=5, title="DOE run", row=len(names), col=1)
    fig.update_layout(bargap=0.15)
    for a in fig.layout.annotations:
        a.update(font=dict(size=12, color=c["text"]), x=0, xanchor="left")
    style(fig, 120 + 160 * len(names), legend=False)
    fig.update_layout(margin=dict(l=8, r=12, t=44, b=8))
    return fig


def contrib_fig(tables: dict[str, pd.DataFrame], order: list[str], xtitle: str) -> go.Figure:
    c = ink()
    first = next(iter(tables.values()))
    avg = {f: float(np.mean([t.set_index("Factor").loc[f, "Contribution (%)"] for t in tables.values()]))
           for f in first["Factor"]}
    factor_order = sorted(avg, key=avg.get)  # smallest first, so the largest sits on top
    fig = go.Figure()
    for name in reversed(list(tables)):
        t = tables[name].set_index("Factor").loc[factor_order]
        vals = t["Contribution (%)"].to_numpy()
        fig.add_trace(go.Bar(
            y=factor_order, x=vals, orientation="h", name=name,
            marker=dict(color=series_color(name, order), line=dict(width=1, color=c["bg"])),
            text=[f"{v:.1f}%" if v >= 5 else "" for v in vals], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>" + name + ": %{x:.2f}%<extra></extra>"))
    n = len(tables)
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06, legend_traceorder="reversed")
    fig.update_xaxes(title=xtitle, ticksuffix="%", rangemode="tozero")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=c["text"]))
    return style(fig, 90 + len(factor_order) * (n * 19 + 20), legend=n > 1)


def main_effects_fig(means: dict[str, dict[str, list[float]]], levels: dict, order: list[str],
                     same_scale: bool) -> go.Figure:
    c = ink()
    keys = rc.FACTOR_KEYS
    fig = make_subplots(rows=3, cols=3, vertical_spacing=0.13, horizontal_spacing=0.06,
                        subplot_titles=[rc.FACTOR_SPECS[k]["label"] for k in keys])
    for idx, k in enumerate(keys):
        r, col = idx // 3 + 1, idx % 3 + 1
        for name, per in means.items():
            fig.add_trace(go.Scatter(
                x=[1, 2, 3, 4, 5], y=per[k], mode="lines+markers", name=name, showlegend=idx == 0,
                legendgroup=name, customdata=levels[k],
                line=dict(width=2, color=series_color(name, order)),
                marker=dict(size=8, color=series_color(name, order), line=dict(width=2, color=c["bg"])),
                hovertemplate=(f"{rc.FACTOR_SPECS[k]['label']} = %{{customdata:.5g}} {rc.FACTOR_SPECS[k]['unit']}"
                               f"<br>{name}: %{{y:.4g}}<extra></extra>")), row=r, col=col)
        fig.update_xaxes(tickvals=[1, 2, 3, 4, 5], row=r, col=col)
        if same_scale:
            fig.update_yaxes(matches="y", row=r, col=col)
    for a in fig.layout.annotations:
        a.update(font=dict(size=12, color=c["text"]))
    style(fig, 700)
    fig.update_layout(margin=dict(l=8, r=12, t=84, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="left", x=0))
    return fig


def gra_rank_fig(ranking: pd.DataFrame, color: str, top: int = 10) -> go.Figure:
    c = ink()
    t = ranking.head(top).iloc[::-1]
    colors = [c["gray"]] * (len(t) - 1) + [color]
    fig = go.Figure(go.Bar(
        y=[f"Run {int(r)}" for r in t["DOE_Run"]], x=t["GRA grade"], orientation="h",
        marker=dict(color=colors, line=dict(width=1, color=c["bg"])),
        text=[f"{v:.3f}" for v in t["GRA grade"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Grade %{x:.4f}<extra></extra>"))
    fig.update_layout(bargap=0.35)
    fig.update_xaxes(title="GRA grade (higher is better)", range=[0, max(1.0, float(t["GRA grade"].max()) * 1.12)])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=c["text"]))
    return style(fig, 70 + 30 * len(t), legend=False)


# ---------------------------------------------------------------------------
# Cached model run
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_model(levels_key: tuple, fixed_key: tuple):
    levels = {k: list(v) for k, v in levels_key}
    return rc.run_model(levels, dict(fixed_key))


def read_table(file) -> pd.DataFrame:
    file.seek(0)
    if file.name.lower().endswith((".xlsx", ".xlsm")):
        return pd.read_excel(file)
    return pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig", encoding_errors="replace")


tab_in, tab_data, tab_res = st.tabs([
    ":material/tune: 1 · Inputs",
    ":material/upload_file: 2 · CFD / test data",
    ":material/insights: 3 · Results",
])

# ===========================================================================
# TAB 1 - INPUTS
# ===========================================================================
with tab_in:
    st.markdown("#### Factor levels")
    st.markdown(
        "Pick **five levels** for each of the nine inputs. The L50 design combines them into "
        "**50 runs** that pair every level of one input with every level of another, so you can rank "
        "the inputs from 50 runs instead of testing all :blue[**1,953,125 combinations**] (5⁹). "
        "Edit any cell in the table; levels must increase from left to right."
    )
    base = pd.DataFrame([
        {"Factor": spec["label"], "Unit": spec["unit"],
         **{f"Level {i + 1}": float(v) for i, v in enumerate(spec["default"])}}
        for spec in rc.FACTOR_SPECS.values()
    ])
    cfg = {
        "Factor": st.column_config.TextColumn("Factor", disabled=True, width="medium"),
        "Unit": st.column_config.TextColumn("Unit", disabled=True, width="small"),
        **{f"Level {i}": st.column_config.NumberColumn(f"Level {i}", format="%.6g", required=True)
           for i in range(1, 6)},
    }
    edited = st.data_editor(base, column_config=cfg, hide_index=True, width="stretch",
                            num_rows="fixed", key=f"levels_{st.session_state.levels_ver}")
    st.caption(
        "Property tables limit the temperatures: water inlet :orange[353.15 to 373.15 K], "
        "air inlet :orange[298.15 to 318.15 K]. Water must stay laminar (:orange[Re below 2300])."
    )

    check_col, fixed_col = st.columns(2, gap="large")

    with fixed_col:
        st.markdown("#### Fixed geometry and material")
        st.caption("These stay the same in all 50 runs.")
        steps = {"tube_length_m": 0.01, "fin_height_m": 0.01,
                 "tube_conductivity_W_mK": 5.0, "fin_conductivity_W_mK": 5.0}
        fixed: dict = {}
        cols = st.columns(2)
        for i, (key, (label, default, unit, typ, helptext)) in enumerate(rc.FIXED_DEFAULTS.items()):
            with cols[i % 2]:
                if typ is int:
                    fixed[key] = st.number_input(f"{label} [{unit}]", min_value=1, value=int(default),
                                                 step=1, help=helptext, key=f"fx_{key}")
                else:
                    fixed[key] = st.number_input(f"{label} [{unit}]", min_value=0.0, value=float(default),
                                                 step=steps[key], format="%g", help=helptext, key=f"fx_{key}")
        if st.button("Reset all inputs to the defaults", icon=":material/restart_alt:"):
            st.session_state.levels_ver += 1
            for k in list(st.session_state):
                if str(k).startswith("fx_"):
                    del st.session_state[k]
            st.rerun()

    levels = {key: [float(edited.iloc[i][f"Level {j}"]) for j in range(1, 6)]
              for i, key in enumerate(rc.FACTOR_KEYS)}
    level_errors, _ = rc.validate_levels(levels)
    fixed_errors = rc.validate_fixed(fixed)
    model_df = codes = None
    invalid: list[str] = []
    failures: list[str] = []
    if not level_errors and not fixed_errors:
        levels_key = tuple((k, tuple(v)) for k, v in levels.items())
        model_df, codes, invalid, failures = cached_model(levels_key, tuple(sorted(fixed.items())))
    model_ok = model_df is not None

    with check_col:
        st.markdown("#### Pre-flight check")
        if level_errors or fixed_errors:
            st.error("Fix these inputs first:\n\n" + "\n".join(f"- {e}" for e in level_errors + fixed_errors),
                     icon=":material/error:")
        elif invalid:
            st.error(f"{len(invalid)} of 50 runs are geometrically impossible:\n\n"
                     + "\n".join(f"- {m}" for m in invalid[:8]) + ("\n- ..." if len(invalid) > 8 else "")
                     + "\n\nChange the levels or the fixed dimensions.", icon=":material/error:")
        elif failures:
            st.error(f"The model cannot calculate {len(failures)} of 50 runs:\n\n"
                     + "\n".join(f"- {m}" for m in failures[:6]) + ("\n- ..." if len(failures) > 6 else "")
                     + "\n\nThe usual cause is water velocity or tube size pushing the water flow out of the "
                       "laminar range.", icon=":material/error:")
        else:
            st.success("All **50 runs** are valid and the model calculated every one.", icon=":material/check_circle:")
            m1, m2 = st.columns(2)
            m1.metric("Water Reynolds number", f"up to {model_df['Rew'].max():.0f}",
                      help="Must stay below 2300 (laminar) for the constant Nusselt number of 4.36 to apply.")
            m2.metric("Air Reynolds number", f"{model_df['Rea'].min():.0f} to {model_df['Rea'].max():.0f}",
                      help="The air-side correlation is valid below 350,000.")
            for w in rc.soft_geometry_warnings(model_df):
                st.warning(w, icon=":material/warning:")

    with st.expander("What each input means"):
        st.dataframe(pd.DataFrame({
            "Input": [s["label"] for s in rc.FACTOR_SPECS.values()],
            "Unit": [s["unit"] for s in rc.FACTOR_SPECS.values()],
            "What it is": [s["help"] for s in rc.FACTOR_SPECS.values()],
        }), hide_index=True, width="stretch")
    if model_ok:
        with st.expander("Show the 50 runs this design uses"):
            st.dataframe(model_df[["DOE_Run"] + rc.FACTOR_KEYS].rename(
                columns={"DOE_Run": "Run", **{k: rc.factor_label(k) for k in rc.FACTOR_KEYS}}),
                hide_index=True, width="stretch", height=320)

# ===========================================================================
# TAB 2 - CFD / TEST DATA
# ===========================================================================
datasets: dict[str, pd.DataFrame] = {}
slot_names: list[str] = []
with tab_data:
    st.markdown("#### CFD or test results (optional)")
    st.markdown(
        "The numerical model needs no data. To compare it with CFD or experiments, provide **one value per DOE run** "
        "(**50 rows**). Use :blue[kelvin] for temperatures and :blue[watts] for heat transfer. "
        "Column names such as :blue[Coolant_Outlet_Temperature_K], :blue[Q_W] and :blue[Effectiveness] are "
        "recognised; any other numeric column (for example air pressure drop) is kept as an extra output."
    )
    if not model_ok:
        st.warning("Fix the inputs in step 1 first. The run list depends on them.", icon=":material/arrow_back:")
    else:
        c1, c2 = st.columns([2, 3], gap="large")
        with c1:
            st.download_button(
                "Download the 50-run sheet (CSV)", rc.run_sheet(model_df).to_csv(index=False),
                file_name="doe_run_sheet.csv", mime="text/csv", icon=":material/download:", width="stretch",
                help="Run each case in CFD or the test rig, fill in the empty result columns, then upload the file.")
        with c2:
            with st.expander("Accepted file layout"):
                st.markdown(
                    "- **Run** column (1 to 50), optional. Without it, rows are read as runs 1 to 50 in order.\n"
                    "- **Coolant_Outlet_Temperature_K**, **Q_W**, **Effectiveness**: any of them.\n"
                    "- Other numeric columns become extra outputs (for example **Air_pressure_drop_Pa**).\n"
                    "- The DOE input columns in the run sheet are ignored, so you can fill in and upload the sheet as is.\n"
                    "- CSV (comma, semicolon or tab separated) or Excel (.xlsx)."
                )
        n_ext = st.segmented_control("How many CFD / test datasets?", list(range(0, MAX_EXTERNAL + 1)), default=1,
                                     format_func=lambda n: "None" if n == 0 else str(n), key="n_ext")
        n_ext = int(n_ext or 0)

        for i in range(n_ext):
            with st.container(border=True):
                h1, h2 = st.columns([1, 2], gap="large")
                name = h1.text_input("Dataset name", value=DEFAULT_NAMES[i], key=f"ds_name_{i}", max_chars=24).strip()
                source = h2.radio("Where do the results come from?", ["Upload a file", "Paste from Excel", "Demo data"],
                                  horizontal=True, key=f"ds_src_{i}")
                slot_names.append(name or DEFAULT_NAMES[i])
                raw = None
                try:
                    if source == "Upload a file":
                        up = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xlsm"], key=f"ds_file_{i}")
                        raw = read_table(up) if up is not None else None
                    elif source == "Paste from Excel":
                        txt = st.text_area("Paste the table with its header row (copy cells in Excel, paste here)",
                                           height=140, key=f"ds_text_{i}")
                        raw = (pd.read_csv(io.StringIO(txt), sep=None, engine="python")
                               if txt.strip() else None)
                    else:
                        st.warning("These are **synthetic numbers** made from the model with an offset and random "
                                   "scatter, so you can try the app. They are not measurements.",
                                   icon=":material/science:")
                        raw = rc.demo_dataset(model_df, "CFD" if i == 0 else "Experiment", seed=7 + i)
                except Exception as exc:  # unreadable file or text
                    st.error(f"Could not read the table: {exc}", icon=":material/error:")
                    raw = None

                if not name:
                    st.error("Give the dataset a name.", icon=":material/error:")
                elif name.casefold() == MODEL.casefold() or name in datasets:
                    st.error(f"'{name}' is already used. Pick a different name.", icon=":material/error:")
                elif raw is not None:
                    try:
                        parsed, notes = rc.parse_external(raw)
                    except ValueError as exc:
                        st.error(str(exc), icon=":material/error:")
                    else:
                        datasets[name] = parsed
                        cols_found = [rc.label_of(k) for k in parsed.columns if k != "DOE_Run"]
                        st.success(f"**{name}**: 50 runs read. Outputs found: {', '.join(cols_found)}.",
                                   icon=":material/check_circle:")
                        for n in notes:
                            st.info(n, icon=":material/info:")
                        with st.expander("Preview"):
                            st.dataframe(parsed, hide_index=True, width="stretch", height=240)
                else:
                    st.caption("Waiting for data. Without it, this dataset is left out of the analysis.")

# ===========================================================================
# SIDEBAR - ANALYSIS CHOICES
# ===========================================================================
selected: list[str] = []
detailed: list[str] = []
gra_cfg = None
with st.sidebar:
    st.markdown("### Choose the analysis")
    if not model_ok:
        st.info("Fix the inputs in step 1 to unlock the choices.", icon=":material/arrow_back:")
    else:
        extra_cols = list(dict.fromkeys(
            k for d in datasets.values() for k in d.columns if k != "DOE_Run" and k not in rc.NUMERICAL_RESPONSE_KEYS))
        in_data = {k for d in datasets.values() for k in d.columns}
        options = rc.NUMERICAL_RESPONSE_KEYS + extra_cols
        default_sel = [k for k in options
                       if k in ("Coolant_Outlet_Temperature_K", "Q_W") or k in extra_cols
                       or (k == "Effectiveness" and k in in_data)]
        selected = st.multiselect(
            "Outputs to analyse", options, default=default_sel, format_func=out_label,
            help="Every chart, table and finding uses only the outputs you pick here.")
        if selected:
            lines = []
            for k in selected:
                where = ([MODEL] if k in rc.NUMERICAL_RESPONSE_KEYS else []) + [n for n, d in datasets.items() if k in d.columns]
                lines.append(f"- **{rc.label_of(k)}**: {', '.join(where)}")
            st.caption("Where each output is available:")
            st.markdown("\n".join(lines))
        else:
            st.warning("Pick at least one output.", icon=":material/warning:")

        detail_options = [MODEL] + list(datasets)
        detailed = st.multiselect(
            "Datasets for factor drivers and GRA", detail_options, default=detail_options,
            help="The comparison always uses every loaded dataset. This choice only controls the factor "
                 "contribution and GRA steps.")

        use_gra = st.toggle(
            "Combine outputs with GRA", value=len(selected) >= 2, disabled=len(selected) < 2,
            help="Grey relational analysis merges several outputs into one grade per run, so you can pick a "
                 "single best run. It needs at least two outputs.")
        if len(selected) < 2:
            st.caption("GRA needs two or more outputs.")
        if use_gra and len(selected) >= 2:
            goals, weights = {}, {}
            with st.expander("GRA goals and weights", expanded=True):
                st.caption("Set what a good result looks like for each output. Weights are relative.")
                for k in selected:
                    st.markdown(f"**{rc.label_of(k)}**")
                    g1, g2 = st.columns(2)
                    goal = g1.selectbox("Goal", ["lower", "higher", "target"],
                                        index=["lower", "higher", "target"].index(rc.default_goal(k)),
                                        key=f"gra_goal_{k}", label_visibility="collapsed")
                    weights[k] = g2.number_input("Weight", min_value=0.01, value=1.0, step=0.1,
                                                 key=f"gra_w_{k}", label_visibility="collapsed")
                    target = None
                    if goal == "target":
                        pool = model_df[k] if k in model_df.columns else next(d[k] for d in datasets.values() if k in d.columns)
                        target = st.number_input(f"Target value ({rc.unit_of(k) or 'same unit'})",
                                                 value=float(pool.mean()), format="%g", key=f"gra_t_{k}")
                    goals[k] = (goal, target)
                total = sum(weights.values())
                st.caption("Normalised weights: " + ", ".join(f"{rc.label_of(k)} {w / total:.2f}" for k, w in weights.items()))
            gra_cfg = {"keys": selected, "goals": goals, "weights": weights}

# ===========================================================================
# TAB 3 - RESULTS
# ===========================================================================
order = [MODEL] + slot_names


def render_summary(analysis: dict, findings: list[dict]) -> None:
    q, tco, eps = model_df["Q_W"], model_df["Coolant_Outlet_Temperature_K"], model_df["Effectiveness"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Heat transfer, Q", f"{q.min():.0f} to {q.max():.0f} W", border=True,
              help="Heat moved from the water to the air. Higher means more cooling.")
    k2.metric("Coolant outlet temperature", f"{tco.min():.1f} to {tco.max():.1f} K", border=True,
              help="Water temperature leaving the radiator. Lower means more heat was removed.")
    k3.metric("Effectiveness", f"{eps.min():.2f} to {eps.max():.2f}", border=True,
              help="Q divided by the most heat the core could possibly move (0 to 1).")
    k4.metric("Datasets compared", f"{len(datasets)}", border=True,
              help="CFD or test datasets loaded in step 2.")
    st.caption("Ranges are for the numerical model across the 50 runs.")

    if not datasets:
        st.info("You are looking at the numerical model alone. Add CFD or test results in step 2 to compare, "
                "or load the demo data there to see the full layout.", icon=":material/info:")

    comp = analysis["comparison"]
    if any(comp.values()):
        st.markdown("##### How well the data agree with the model")
        badge = {"Good": ":green-badge[:material/check: Good]", "Moderate": ":orange-badge[:material/remove: Moderate]",
                 "Poor": ":red-badge[:material/close: Poor]", "n/a": ":gray-badge[n/a]"}
        rows = ["| Dataset | Output | Mean error (data − model) | NRMSE | Correlation r | Agreement |",
                "|---|---|---:|---:|---:|---|"]
        for ds, per in comp.items():
            for k, m in per.items():
                nr = "n/a" if m["nrmse_pct"] is None else f"{m['nrmse_pct']:.1f}%"
                rows.append(f"| {ds} | {rc.label_of(k)} | {num(m['bias'])} {rc.unit_of(k)} | {nr} | {num(m['r'], 3)} | "
                            f"{badge[rc.agreement_rating(m['nrmse_pct'])]} |")
        st.markdown("\n".join(rows))
        st.caption("NRMSE is the root-mean-square error as a percentage of the data's range. "
                   "Under 5% reads as good, 5 to 15% as moderate, above 15% as poor.")
        for f in (x for x in findings if x["section"] == "compare"):
            st.markdown(f"- **{f['title']}.** {f['text']}")

    drivers = [x for x in findings if x["section"] == "drivers"]
    if drivers:
        st.markdown("##### What drives each output")
        for f in drivers:
            st.markdown(f"- **{f['title']}.** {f['text']}")

    for f in (x for x in findings if x["section"] == "gra"):
        st.markdown("##### Best run")
        st.success(f"**{f['title']}**\n\n{f['text']}", icon=":material/emoji_events:")

    st.markdown("##### Cautions")
    for f in (x for x in findings if x["section"] == "caution"):
        (st.warning if f["kind"] == "warning" else st.info)(f"**{f['title']}**\n\n{f['text']}",
                                                            icon=":material/warning:" if f["kind"] == "warning" else ":material/info:")
    with st.expander("What do the outputs mean?"):
        for k in selected:
            spec = rc.RESPONSE_SPECS.get(k)
            if spec:
                goal = {"lower": ":blue[lower is better]", "higher": ":blue[higher is better]"}[spec["goal"]]
                st.markdown(f"**{out_label(k)}** ({goal}). {spec['help']}")
            else:
                st.markdown(f"**{rc.label_of(k)}**. Extra output from your uploaded data. "
                            f"Suggested GRA goal: :blue[{rc.default_goal(k)} is better].")


def render_compare(analysis: dict) -> None:
    comp = analysis["comparison"]
    keys = [k for k in dict.fromkeys(k for per in comp.values() for k in per)]
    if not keys:
        st.info("Load a CFD or test dataset in step 2 that contains a selected output to see the comparison.",
                icon=":material/info:")
        return
    key = st.radio("Output to compare", keys, format_func=out_label, horizontal=True)
    sets = {n: d for n, d in datasets.items() if key in d.columns}
    runs = model_df["DOE_Run"].to_numpy()
    mv = model_df[key].to_numpy()

    st.markdown("##### Error metrics")
    rows = []
    for ds in sets:
        m = comp[ds][key]
        rows.append({"Dataset": ds, "Mean error": m["bias"], "MAE": m["mae"], "RMSE": m["rmse"],
                     "Max |error|": m["max_abs_error"], "NRMSE (%)": m["nrmse_pct"],
                     "Correlation r": m["r"], "Agreement": rc.agreement_rating(m["nrmse_pct"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4g")
                                for c in ("Mean error", "MAE", "RMSE", "Max |error|", "NRMSE (%)", "Correlation r")})
    st.caption(f"Errors are data minus model in {rc.unit_of(key) or 'the output unit'}. A positive mean error means "
               "the CFD or test value is higher than the model.")

    st.markdown("##### Model against data, run by run")
    reading = (":blue[Points on the line] mean the two agree. :orange[Points above it] mean the data read higher "
               "than the model. A tight cloud that sits off the line is a steady offset; a wide cloud means the "
               "two disagree about how the runs differ.")
    cols = st.columns(max(len(sets), 2))
    if len(sets) > 1:
        st.markdown(reading)
    for col, (ds, d) in zip(cols, sets.items()):
        with col:
            st.markdown(f"**{ds}**")
            show(parity_fig(mv, d[key].to_numpy(), runs, key, ds, series_color(ds, order)))
    if len(sets) == 1:
        with cols[1]:
            st.markdown("**How to read this**")
            st.markdown(reading)

    st.markdown("##### All sources by DOE run")
    series = {MODEL: mv, **{ds: d[key].to_numpy() for ds, d in sets.items()}}
    show(runs_fig(series, runs, key, order))
    st.markdown("##### Error by run")
    show(error_fig({ds: d[key].to_numpy() - mv for ds, d in sets.items()}, runs, key, order))


def render_drivers(analysis: dict) -> None:
    anova = analysis["anova"]
    keys = list(dict.fromkeys(k for per in anova.values() for k in per))
    if not keys:
        st.info("Pick at least one output and one dataset in the sidebar.", icon=":material/info:")
        return
    st.markdown(
        "Each bar is the share of an output's total variation that one input explains on its own "
        "(:blue[main-effect contribution]). The bars plus the :orange[interaction share] add to 100%.")
    key = st.radio("Output", keys, format_func=out_label, horizontal=True, key="drv_out")
    per = {n: v[key] for n, v in anova.items() if key in v}
    tables = {n: t for n, (t, _) in per.items()}
    show(contrib_fig(tables, order, f"Contribution to variation in {rc.label_of(key).lower()}"))

    cols = st.columns(len(per))
    for col, (n, (t, meta)) in zip(cols, per.items()):
        col.metric(n, f"{meta['r2_pct']:.1f}% explained",
                   help=f"Main effects explain {meta['r2_pct']:.1f}% of the variation. The other "
                        f"{meta['residual_pct']:.1f}% comes from interactions between inputs or model mismatch.")
    st.caption("Below about 90% explained, interactions matter and the ranking is only a guide.")

    pivot = pd.DataFrame({n: t.set_index("Factor")["Contribution (%)"] for n, t in tables.items()})
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index].reset_index()
    st.dataframe(pivot, hide_index=True, width="stretch", column_config={
        n: st.column_config.ProgressColumn(n, min_value=0, max_value=100, format="%.1f%%") for n in tables})

    with st.expander("How the output changes across the five levels of each input"):
        st.caption("Average output at each level of the input. A steep line means a strong effect; a flat line means little.")
        same = st.checkbox("Use the same vertical scale for every input", value=True)
        show(main_effects_fig({n: meta["level_means"] for n, (_, meta) in per.items()}, levels, order, same))
    with st.expander("Detailed ANOVA tables"):
        for n, (t, meta) in per.items():
            st.markdown(f"**{n}**")
            st.dataframe(t.drop(columns=["key"]), hide_index=True, width="stretch")


def render_gra(analysis: dict) -> None:
    gra = analysis["gra"]
    if gra_cfg is None:
        st.info("Turn on **Combine outputs with GRA** in the sidebar (needs two or more outputs).", icon=":material/info:")
        return
    if not gra:
        st.warning("GRA could not run for the chosen datasets. See the cautions in the summary.", icon=":material/warning:")
        return
    st.markdown(
        "Grey relational analysis turns each output into a 0 to 1 score (1 = the goal you set), merges the scores "
        "with your weights, and gives every run one :blue[grade]. The run with the highest grade is the best compromise "
        "among the 50 tested settings.")
    first = next(iter(gra.values()))
    linked = [k for k in first["cfg"]["keys"] if k in rc.NUMERICAL_RESPONSE_KEYS]
    if len(linked) >= 2:
        st.info("**Linked outputs.** " + ", ".join(rc.label_of(k) for k in linked) + " are tied together by the heat "
                "balance, so scoring them side by side counts the same physics more than once. Lower their weights "
                "or add an independent output such as pressure drop to get a fairer grade.",
                icon=":material/link:")
    st.dataframe(pd.DataFrame({
        "Output": [out_label(k) for k in first["cfg"]["keys"]],
        "Goal": [f"target {first['cfg']['goals'][k][1]:g}" if first["cfg"]["goals"][k][0] == "target"
                 else first["cfg"]["goals"][k][0] for k in first["cfg"]["keys"]],
        "Weight": [first["cfg"]["weights"][k] for k in first["cfg"]["keys"]]}),
        hide_index=True, width="stretch", column_config={"Weight": st.column_config.NumberColumn(format="%.2f")})
    st.caption("If a dataset lacks one of these outputs, that output is left out for that dataset and the "
               "remaining weights are rescaled to add up to 1.")

    best = {n: int(g["ranking"].iloc[0]["DOE_Run"]) for n, g in gra.items()}
    if len(set(best.values())) > 1:
        st.warning("The datasets pick **different best runs**: " + "; ".join(f"{n} → Run {r}" for n, r in best.items())
                   + ". Check which source you trust for the final choice.", icon=":material/compare_arrows:")
    elif len(best) > 1:
        st.success(f"All datasets agree: :green[**Run {next(iter(best.values()))}**] is the best run.",
                   icon=":material/check_circle:")

    pick = st.radio("Dataset", list(gra), horizontal=True, key="gra_pick") if len(gra) > 1 else next(iter(gra))
    g = gra[pick]
    run = best[pick]
    frame = model_df if pick == MODEL else datasets[pick]
    row_in = model_df.loc[model_df["DOE_Run"] == run].iloc[0]
    row_out = frame.loc[frame["DOE_Run"] == run].iloc[0]
    grade = float(g["grades"][g["best_index"]])
    runner = float(g["ranking"].iloc[1]["GRA grade"]) if len(g["ranking"]) > 1 else grade

    m1, m2, m3 = st.columns(3)
    m1.metric("Best run", f"Run {run}", border=True)
    m2.metric("GRA grade", f"{grade:.3f}", delta=f"{grade - runner:+.3f} vs second", border=True)
    m3.metric("Main driver of the grade", g["table"].iloc[0]["Factor"].split(" (")[0], border=True,
              help=f"{g['table'].iloc[0]['Contribution (%)']:.1f}% of the variation in the grade.")

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(f"**Settings of Run {run}**")
        st.dataframe(pd.DataFrame({
            "Input": [rc.FACTOR_SPECS[k]["label"] for k in rc.FACTOR_KEYS],
            "Value": [row_in[k] for k in rc.FACTOR_KEYS],
            "Unit": [rc.FACTOR_SPECS[k]["unit"] for k in rc.FACTOR_KEYS]}),
            hide_index=True, width="stretch", column_config={"Value": st.column_config.NumberColumn(format="%.6g")})
    with b:
        st.markdown(f"**Outputs of Run {run} ({pick})**")
        st.dataframe(pd.DataFrame({
            "Output": [rc.label_of(k) for k in g["cfg"]["keys"]],
            "Value": [row_out[k] for k in g["cfg"]["keys"]],
            "Unit": [rc.unit_of(k) for k in g["cfg"]["keys"]]}),
            hide_index=True, width="stretch", column_config={"Value": st.column_config.NumberColumn(format="%.5g")})
        st.markdown("**Top 10 runs**")
        show(gra_rank_fig(g["ranking"], series_color(pick, order)))

    st.markdown("##### What moves the grade")
    show(contrib_fig({n: x["table"] for n, x in gra.items()}, order, "Contribution to variation in the GRA grade"))
    with st.expander("Full ranking of all 50 runs"):
        st.dataframe(g["ranking"].assign(Rank=np.arange(1, len(g["ranking"]) + 1))[["Rank", "DOE_Run", "GRA grade"]],
                     hide_index=True, width="stretch", height=360)


def render_export(analysis: dict, findings: list[dict]) -> None:
    st.markdown("##### Download")
    d1, d2 = st.columns(2)
    d1.download_button("Everything as one Excel workbook", rc.export_workbook(model_df, datasets, analysis, findings),
                       file_name="radiator_doe_results.xlsx", icon=":material/table:", width="stretch",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2.download_button("Numerical model table (CSV)", model_df.to_csv(index=False), file_name="numerical_model_results.csv",
                       mime="text/csv", icon=":material/download:", width="stretch")
    st.markdown("##### Numerical model, all 50 runs")
    st.dataframe(model_df, hide_index=True, width="stretch", height=420)
    st.caption("Rew and Rea are the water and air Reynolds numbers, UA the overall conductance, NTU the number of "
               "transfer units, Cr the heat-capacity-rate ratio. The energy balance error should be near zero.")


with tab_res:
    if not model_ok:
        st.warning("Fix the inputs in step 1. Results appear once all 50 runs are valid.", icon=":material/arrow_back:")
    elif not selected:
        st.info("Pick at least one output in the sidebar.", icon=":material/arrow_back:")
    else:
        model_keys_sel = [k for k in selected if k in rc.NUMERICAL_RESPONSE_KEYS]
        comparison_keys = [k for k in model_keys_sel if any(k in d.columns for d in datasets.values())]
        model_only = [k for k in model_keys_sel if k not in comparison_keys]
        extra_sel = [k for k in selected if k not in rc.NUMERICAL_RESPONSE_KEYS]
        analysis = rc.analyze(model_df, codes, datasets, comparison_keys, model_only, extra_sel, detailed, gra_cfg)
        findings = rc.build_findings(model_df, analysis, comparison_keys)
        st.caption("Change the outputs, datasets and GRA goals in the sidebar. Everything below updates.")
        names = ["Summary", "Compare", "Factor drivers", "Best run (GRA)", "Tables and export"]
        t_sum, t_cmp, t_drv, t_gra, t_exp = st.tabs(names)
        with t_sum:
            render_summary(analysis, findings)
        with t_cmp:
            render_compare(analysis)
        with t_drv:
            render_drivers(analysis)
        with t_gra:
            render_gra(analysis)
        with t_exp:
            render_export(analysis, findings)

st.divider()
st.caption("The numerical model is a correlation-based estimate (laminar water flow with Nu = 4.36, a flat-plate "
           "air-side correlation and the effectiveness-NTU method for crossflow). Contribution percentages describe "
           "main effects over the levels you entered, not a physical law.")
