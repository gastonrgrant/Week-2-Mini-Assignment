import pandas as pd

from cbb_analysis import (
    add_win_percentage,
    get_top_teams_by_year,
    load_data,
    train_winning_percentage_model,
)


def make_test_data():
    """Create a small dataset for testing."""
    return pd.DataFrame(
        {
            "YEAR": [2020, 2020, 2021, 2021, 2022, 2022],
            "TEAM": ["A", "B", "C", "D", "E", "F"],
            "W": [25, 15, 28, 12, 24, 16],
            "G": [30, 30, 30, 30, 30, 30],
            "EFG_O": [0.55, 0.45, 0.60, 0.42, 0.58, 0.44],
            "TOR": [15, 20, 14, 22, 16, 21],
            "ORB": [35, 25, 38, 23, 34, 24],
            "FTR": [0.20, 0.15, 0.22, 0.14, 0.21, 0.16],
            "DRB": [30, 22, 33, 21, 31, 23],
            "TORD": [18, 12, 20, 10, 19, 11],
            "EFG_D": [0.42, 0.52, 0.40, 0.55, 0.43, 0.53],
            "FTRD": [0.15, 0.22, 0.14, 0.24, 0.16, 0.23],
        }
    )


def test_add_win_percentage():
    data = make_test_data()

    result = add_win_percentage(data)

    assert "WIN_PCT" in result.columns
    assert result.loc[0, "WIN_PCT"] == 25 / 30
    assert "WIN_PCT" not in data.columns


def test_model_trains_and_returns_metrics():
    data = add_win_percentage(make_test_data())

    model, metrics, predictions, actual = train_winning_percentage_model(data)

    assert len(predictions) == len(actual)
    assert "mae" in metrics
    assert "mse" in metrics
    assert "r2" in metrics
    # mae cannot be negative
    assert metrics["mae"] >= 0


def test_top_teams_returns_correct_number_per_year():
    data = add_win_percentage(make_test_data())

    model, _, _, _ = train_winning_percentage_model(data)
    result = get_top_teams_by_year(data, model, number_of_teams=1)
    #make sure each year has a predicted top team
    assert len(result) == data["YEAR"].nunique()
    # check that the number of teams for each year is one
    assert result.groupby("YEAR").size().max() == 1
    assert "RANK" in result.columns
    # asser that the top team is ranked 1
    assert result["RANK"].eq(1).all()


def test_load_data(tmp_path):
    data = make_test_data()
    #use tmp_path so pytest will automatically create a test folder
    filepath = tmp_path / "test_cbb.csv"
    data.to_csv(filepath, index=False)

    loaded_data = load_data(filepath)

    assert len(loaded_data) == len(data)
    assert list(loaded_data.columns) == list(data.columns)