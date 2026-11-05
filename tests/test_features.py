import pandas as pd

from src.features import add_time_features


def test_time_features_saturday_afternoon():
    out = add_time_features(pd.DataFrame({"datetime": ["2019-07-06 15:00:00"]}))
    assert out.loc[0, "hour"] == 15
    assert out.loc[0, "dayofweek"] == 5  # Saturday (Monday = 0)
    assert out.loc[0, "month"] == 7
    assert out.loc[0, "is_weekend"] == 1


def test_time_features_monday_is_weekday():
    out = add_time_features(pd.DataFrame({"datetime": ["2019-07-01 00:00:00"]}))
    assert out.loc[0, "dayofweek"] == 0
    assert out.loc[0, "is_weekend"] == 0


def test_time_features_do_not_modify_input():
    df = pd.DataFrame({"datetime": ["2019-07-01 00:00:00"]})
    add_time_features(df)
    assert list(df.columns) == ["datetime"]
