import pandas as pd
import pytest

from src.metrics import MetricAccumulator


def synthetic_drive() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sim Time": [10.0, 11.0, 13.0],
            "Velocity": [2.0, 4.0, 6.0],
        }
    )


def test_basic_values_from_synthetic_frame() -> None:
    frame = synthetic_drive()
    accumulator = MetricAccumulator()
    accumulator.update(frame, raw_row_count=5)
    summary = accumulator.finalize()

    assert summary["raw_rows"] == 5
    assert summary["driving_rows"] == 3
    assert summary["duration"] == pytest.approx(3.0)
    assert summary["mean_velocity"] == pytest.approx(4.0)
    assert summary["max_velocity"] == pytest.approx(6.0)


def test_chunked_accumulator_matches_single_frame() -> None:
    frame = synthetic_drive()
    chunked = MetricAccumulator()
    chunked.update(frame.iloc[:2], raw_row_count=2)
    chunked.update(frame.iloc[2:], raw_row_count=1)

    complete = MetricAccumulator()
    complete.update(frame, raw_row_count=3)

    assert chunked.finalize() == complete.finalize()
