import numpy as np
import pandas as pd
import pytest

from src.metrics import regression_metrics, metrics_by_group


def test_perfect_prediction_is_zero():
    m = regression_metrics([-3.0, -1.0, 0.5], [-3.0, -1.0, 0.5])
    assert m["mae_log"] == 0 and m["mse_log"] == 0 and m["rmse_log"] == 0 and m["mdape_pct"] == 0


def test_hand_computed_values():
    # errors: +1, -1, +2  -> MAE 4/3, MSE 6/3 = 2, RMSE sqrt(2)
    m = regression_metrics([0.0, 0.0, 0.0], [1.0, -1.0, 2.0])
    assert m["mae_log"] == pytest.approx(4 / 3)
    assert m["mse_log"] == pytest.approx(2.0)
    assert m["rmse_log"] == pytest.approx(np.sqrt(2))
    # relative errors on seconds: |10-1|=9, |0.1-1|=0.9, |100-1|=99 -> median 9 -> 900 %
    assert m["mdape_pct"] == pytest.approx(900.0)


def test_log_error_is_relative():
    # a factor-10 miss costs the same at 1 ms and at 10 s
    a = regression_metrics([-3.0], [-2.0])["mae_log"]
    b = regression_metrics([1.0], [2.0])["mae_log"]
    assert a == pytest.approx(b) == pytest.approx(1.0)


def test_metrics_by_group():
    df = pd.DataFrame({"slice_type": ["eMBB", "eMBB", "mMTC"], "y": [0.0, 0.0, 0.0], "pred": [1.0, 3.0, 0.0]})
    out = metrics_by_group(df, ["slice_type"]).set_index("slice_type")
    assert out.loc["eMBB", "mae_log"] == pytest.approx(2.0)
    assert out.loc["mMTC", "mae_log"] == 0.0
    assert out.loc["ALL", "n"] == 3 and out.loc["ALL", "mae_log"] == pytest.approx(4 / 3)
