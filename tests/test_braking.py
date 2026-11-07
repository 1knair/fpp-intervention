import numpy as np
import pandas as pd
import pytest

from src.representation.braking_features import braking_signals, find_braking_curves, summarize_braking


def drive(step=0.02):
    times = np.arange(0, 8, step)
    return pd.DataFrame({"Sim Time": times, "Velocity": np.full(len(times), 12.),
                         "Acceleration": np.zeros(len(times)), "Brake Pedal Force": np.zeros(len(times))})


def add_press(telem, start, end, force=40):
    active = telem["Sim Time"].ge(start) & telem["Sim Time"].lt(end)
    telem.loc[active, "Brake Pedal Force"] = force
    telem.loc[active, "Acceleration"] = -2


def test_press_has_exact_bounds_and_is_one_curve_not_many_samples():
    telem = drive()
    add_press(telem, 1, 3)
    curves = find_braking_curves(telem)
    assert len(curves) == 1
    curve = curves.iloc[0]
    assert curve.start_row == 50 and curve.end_row == 151
    assert curve.start_time_s == pytest.approx(1)
    assert curve.end_time_s == pytest.approx(3)
    assert curve.peak_decel_mps2 == pytest.approx(2)
    assert np.isfinite(curve.jerk_rms_mps3)
    summary = summarize_braking(curves, telem)
    assert summary["brake_episodes_per_km"] == pytest.approx(1 / summary["brake_curve_exposure_km"])


@pytest.mark.parametrize("release_gap, expected", [(0.1, 1), (0.4, 2)])
def test_short_pedal_release_merges_but_separate_presses_remain_separate(release_gap, expected):
    telem = drive()
    add_press(telem, 1, 2)
    add_press(telem, 2 + release_gap, 4)
    assert len(find_braking_curves(telem)) == expected


def test_short_noise_stationary_press_truncation_and_gaps_are_excluded():
    telem = drive()
    add_press(telem, 0, 0.8)            # unknown onset
    add_press(telem, 1, 1.2)            # short tap
    add_press(telem, 2, 2.6, force=3)   # below peak threshold, separated from next press
    add_press(telem, 3, 4)
    telem.loc[telem["Sim Time"].between(3, 4), "Velocity"] = 0
    add_press(telem, 5, 6.5)
    telem.loc[telem["Sim Time"].ge(5.5), "Sim Time"] += 5
    add_press(telem, 12, 13)            # unknown release at file end
    assert find_braking_curves(telem).empty


def test_brake_curve_ends_at_stop_without_including_stationary_hold():
    telem = drive()
    add_press(telem, 1, 6)
    telem.loc[telem["Sim Time"].ge(3), "Velocity"] = 0
    curves = find_braking_curves(telem)
    assert len(curves) == 1
    assert curves.iloc[0].end_time_s == pytest.approx(3)
    assert curves.iloc[0].end_reason == "stopped"


def test_smooth_and_abrupt_curves_have_different_rise_times_and_jerk():
    smooth, abrupt = drive(), drive()
    times = smooth["Sim Time"].to_numpy()
    profile = np.where((times >= 1) & (times < 4), np.minimum((times - 1) * 100, 100), 0)
    smooth["Brake Pedal Force"] = profile
    smooth["Acceleration"] = -profile / 20
    add_press(abrupt, 1, 4, force=100)
    abrupt.loc[abrupt["Brake Pedal Force"].gt(0), "Acceleration"] = -5
    smooth_curve = find_braking_curves(smooth).iloc[0]
    abrupt_curve = find_braking_curves(abrupt).iloc[0]
    assert smooth_curve.rise_time_s > abrupt_curve.rise_time_s * 3
    assert smooth_curve.jerk_rms_mps3 < abrupt_curve.jerk_rms_mps3
    assert smooth_curve.peak_decel_mps2 == pytest.approx(abrupt_curve.peak_decel_mps2)


def test_smoothing_and_jerk_do_not_bridge_gaps():
    telem = drive()
    later = telem["Sim Time"].ge(4)
    telem.loc[~later, "Acceleration"] = -2
    telem.loc[later, "Acceleration"] = 2
    telem.loc[later, "Sim Time"] += 5
    signals = braking_signals(telem)
    assert signals.jerk_mps3.dropna().abs().max() < 1e-9


def test_no_brakes_means_zero_rate_but_missing_curve_shape():
    telem = drive()
    summary = summarize_braking(find_braking_curves(telem), telem)
    assert summary["brake_episodes_per_km"] == 0
    assert pd.isna(summary["brake_rise_time_median_s"])
    assert pd.isna(summary["brake_jerk_rms_p90_mps3"])
