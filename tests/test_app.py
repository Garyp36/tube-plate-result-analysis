"""Smoke tests that run the whole Streamlit script headlessly. Run with:  pytest -q"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture()
def app():
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    return at


def test_loads_without_errors(app):
    assert not app.exception
    assert not app.error
    assert any("All **50 runs** are valid" in s.value for s in app.success)


def test_demo_data_unlocks_comparison_and_gra(app):
    app.radio(key="ds_src_0").set_value("Demo data").run()
    assert not app.exception
    outputs = [m for m in app.multiselect if m.label == "Outputs to analyse"][0]
    assert "Air pressure drop Pa" in outputs.options
    assert any("Best DOE run by combined GRA grade" in s.value for s in app.success)


def test_two_datasets_with_demo_data(app):
    app.radio(key="ds_src_0").set_value("Demo data").run()
    app.segmented_control(key="n_ext").set_value(2).run()
    app.radio(key="ds_src_1").set_value("Demo data").run()
    assert not app.exception
    assert not app.error


def test_duplicate_dataset_name_is_flagged(app):
    app.radio(key="ds_src_0").set_value("Demo data").run()
    app.segmented_control(key="n_ext").set_value(2).run()
    app.text_input(key="ds_name_1").set_value("CFD").run()
    app.radio(key="ds_src_1").set_value("Demo data").run()
    assert any("already used" in e.value for e in app.error)
