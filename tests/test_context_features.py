import numpy as np
import pandas as pd
import pytest

from src.representation.braking_features import find_braking_curves
from src.representation.event_features import calc_event_features, find_hump_candidates, lane_closure_windows


def drive():
    times = np.arange(0, 20, .02)
    return pd.DataFrame({'Sim Time': times, 'Velocity': 5., 'Acceleration': 0.,
        'Brake Pedal Force': 0., 'Speed Limit': 25., 'Signs': 0., 'Events': 0.,
        'Z Position': -.65, 'X Position': times * 5, 'Y Position': 0., 'Lane Number': 1.})


def test_two_vertical_crests_produce_two_ordered_crossing_speeds():
    telem = drive()
    t = telem['Sim Time']
    telem['Z Position'] += .16 * np.exp(-((t - 5) / .15) ** 2) + .16 * np.exp(-((t - 15) / .15) ** 2)
    features, humps, _ = calc_event_features(telem, find_braking_curves(telem))
    assert humps.time_s.tolist() == pytest.approx([5, 15])
    assert features['hump1_crossing_mph'] == pytest.approx(5 / .44704)
    assert features['hump2_crossing_mph'] == pytest.approx(5 / .44704)
    assert features['humps_over15_count'] == 0


def test_unknown_or_ambiguous_humps_are_missing_not_safe_zero():
    telem = drive()
    for at in [3, 8, 15]:
        telem['Z Position'] += .16 * np.exp(-((telem['Sim Time'] - at) / .15) ** 2)
    features, _, _ = calc_event_features(telem, find_braking_curves(telem))
    assert features['hump_candidate_count'] == 3
    assert pd.isna(features['hump1_crossing_mph'])
    assert pd.isna(features['humps_over15_count'])
    telem['Z Position'] = np.nan
    assert find_hump_candidates(telem).empty


def test_crest_crossing_a_clock_gap_is_not_detected():
    telem = drive()
    telem['Z Position'] += .16 * np.exp(-((telem['Sim Time'] - 5) / .15) ** 2)
    telem.loc[telem['Sim Time'].ge(5), 'Sim Time'] += 10
    assert find_hump_candidates(telem).empty


def test_unknown_crossing_speed_is_not_an_observed_hump():
    telem = drive()
    telem['Z Position'] += .16 * np.exp(-((telem['Sim Time'] - 5) / .15) ** 2)
    telem.loc[telem['Sim Time'].eq(5), 'Velocity'] = np.nan
    assert find_hump_candidates(telem).empty


@pytest.mark.parametrize('event_code', [4, 5])
def test_lane_locator_uses_sign_context_instead_of_assuming_one_event_code(event_code):
    telem = drive()
    telem.loc[telem['Sim Time'].eq(2), 'Signs'] = -6
    telem.loc[telem['Sim Time'].between(5, 10, inclusive='left'), 'Events'] = event_code
    telem.loc[telem['Sim Time'].eq(12), 'Signs'] = -7
    dt = telem['Sim Time'].shift(-1) - telem['Sim Time']
    assert lane_closure_windows(telem, dt) == [(250, 500)]
    telem.loc[300:, 'Sim Time'] += 5
    dt = telem['Sim Time'].shift(-1) - telem['Sim Time']
    assert lane_closure_windows(telem, dt) == []


def test_opposite_direction_lane_flip_is_not_counted_as_lane_change():
    telem = drive()
    telem.loc[100, 'Signs'] = -6
    telem.loc[250:499, 'Events'] = 4
    telem.loc[600, 'Signs'] = -7
    telem.loc[150:, 'Lane Number'] = -1
    features, _, _ = calc_event_features(telem, find_braking_curves(telem))
    assert features['lane_change_count'] == 0
    assert pd.isna(features['lane_last_change_lead_s'])
