# Radiator DOE Analyzer

A Streamlit app for crossflow-radiator studies. It builds a 50-run Taguchi L50 design over nine inputs, calculates every run with a correlation-based numerical model, compares the results with your CFD or test data, ranks the inputs that drive each output, and picks the best run with grey relational analysis (GRA).

It is the interactive version of the original notebook script. The model, the L50 design, the ANOVA and the GRA are the same; the app adds checks, charts and plain-language findings.

## What you get

| Step | What happens |
|---|---|
| 1. Inputs | Edit five levels for each of nine inputs. A pre-flight check says whether all 50 runs are valid and warns about odd geometry. |
| 2. CFD / test data | Download the 50-run sheet, fill in your results, upload it (CSV or Excel), paste from Excel, or load synthetic demo data. Up to three datasets. |
| Sidebar | Choose which outputs to analyse, which datasets get the factor and GRA steps, and the GRA goals and weights. |
| 3. Results | Summary with findings, model-vs-data comparison, factor drivers, best run, and a full Excel export. |

Colored text in the app marks what matters: blue for key numbers and terms, orange for offsets and cautions, green for good agreement or the best run, violet for the leading factor.

## Files

```
app.py                  the Streamlit interface
radiator_core.py        model, L50 design, ANOVA, GRA, parsing, findings (no Streamlit code)
requirements.txt        Python packages
.streamlit/config.toml  theme (light and dark)
tests/                  pytest checks for the core and the app
.github/workflows/      runs the tests on every push
```

## Run it on your computer

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run the tests with `pip install pytest` then `pytest -q`.

## Put it on GitHub and publish it

1. Create a new repository on github.com (for example `radiator-doe-analyzer`). Public repositories deploy for free.
2. Upload every file and folder from this project, keeping the structure above. On the repository page choose **Add file > Upload files** and drag the whole folder in. Make sure `.streamlit/config.toml` and `.github/workflows/tests.yml` are included (folders that start with a dot are hidden on some systems).
3. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
4. Choose **Create app**, pick your repository, set the branch to `main` and the main file to `app.py`, then deploy.
5. The first build takes a few minutes. Every later push to GitHub updates the app by itself.

## File format for CFD or test results

One row per DOE run, 50 rows in total. Temperatures in kelvin, heat in watts.

| Column | Meaning |
|---|---|
| `Run` | 1 to 50, optional. Without it, rows are read in order. |
| `Coolant_Outlet_Temperature_K` | Water temperature leaving the radiator. Also accepted: `Tco`, `Water outlet temperature`. |
| `Q_W` | Heat transfer rate. Also accepted: `Q`, `Heat transfer rate`, `Heat duty`. |
| `Effectiveness` | Optional, between 0 and 1. |
| any other numeric column | Kept as an extra output, for example `Air_pressure_drop_Pa`. |

The **Download the 50-run sheet** button in the app gives you a file with the settings of all 50 runs and empty result columns, so you can fill it in and upload it unchanged.

## How to read the results

- **NRMSE** is the root-mean-square error between data and model as a percentage of the data range. Under 5% reads as good, 5 to 15% as moderate, above 15% as poor. These bands are a rule of thumb, not a standard.
- **Mean error** is data minus model. Positive means the CFD or test value is higher.
- **Contribution (%)** is the share of an output's variation that one input explains on its own. The leftover share is interaction between inputs or model mismatch. Below about 90% explained, treat the ranking as a guide.
- **GRA grade** merges several outputs into one score per run. The best run is the best of the 50 sampled settings, not a continuous optimum.

## Things worth knowing about the model

These came up while checking the code against the default inputs:

- With the default levels the water side has the smaller heat-capacity rate in all 50 runs (water velocity is very low), so the coolant limits heat transfer and air-side inputs matter less. The app flags this.
- 20 of the 50 default runs have a fin narrower than the tube (fin width below tube width). The model does not check this and still calculates them. The app warns about it.
- Coolant outlet temperature, heat transfer rate and effectiveness are linked by the heat balance. Using all three in GRA counts the same physics several times, so lower their weights or add an independent output such as pressure drop.
- Temperature property tables only cover 353.15 to 373.15 K (water) and 298.15 to 318.15 K (air). Water must stay laminar (Re below 2300) for the constant Nusselt number of 4.36 to hold.
- The demo data are synthetic. They are made from the model with an offset and random scatter.

## Checked against the original script

For the default inputs, this project's model outputs, ANOVA contributions, GRA grades and error metrics match the original notebook code to machine precision. The tests in `tests/` cover the design, the model, the ANOVA, the GRA, file parsing, the Excel export and the app itself.
