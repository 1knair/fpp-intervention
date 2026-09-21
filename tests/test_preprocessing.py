import pandas as pd
import pytest

from src.preprocessing import filter_driving_rows


def test_filter_driving_rows_keeps_only_gear_three_and_preserves_input() -> None:
    frame = pd.DataFrame({"Gear": [1, 3, "3", 2], "Velocity": [0, 10, 20, 5]})
    original = frame.copy(deep=True)

    filtered = filter_driving_rows(frame)

    assert filtered["Velocity"].tolist() == [10, 20]
    assert filtered.index.tolist() == [1, 2]
    pd.testing.assert_frame_equal(frame, original)


def test_filter_driving_rows_requires_gear() -> None:
    with pytest.raises(ValueError, match="Gear"):
        filter_driving_rows(pd.DataFrame({"Velocity": [10]}))
