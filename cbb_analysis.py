import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Dean Oliver's Four Factors, measured on offense and on defense.
FEATURES = ["EFG_O", "TOR", "ORB", "FTR", "DRB", "TORD", "EFG_D", "FTRD"]

# Positive means a higher value should go with a higher winning percentage.
EXPECTED_SIGN = {
    "EFG_O": 1,
    "TOR": -1,
    "ORB": 1,
    "FTR": 1,
    "DRB": 1,
    "TORD": 1,
    "EFG_D": -1,
    "FTRD": -1,
}


def load_data(filepath):
    """Load the college basketball CSV."""
    return pd.read_csv(filepath)


def clean_team_data(data):
    """Drop team-seasons that cannot have a real winning percentage.

    A row is removed when games played is not positive, or when wins are
    greater than games played. Those rows are recording errors.
    """
    if "W" not in data.columns or "G" not in data.columns:
        raise KeyError("W and G are required to clean team results.")

    valid_games = data["G"] > 0
    valid_wins = data["W"] <= data["G"]
    return data.loc[valid_games & valid_wins].copy()


def add_win_percentage(data):
    """Add WIN_PCT without changing the original table."""
    if (data["G"] <= 0).any():
        raise ValueError("Games played must be greater than zero.")

    result = data.copy()
    result["WIN_PCT"] = result["W"] / result["G"]
    return result


def select_features(data):
    """Select the eight Four Factors and the winning-percentage target."""
    missing = [column for column in FEATURES if column not in data.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    if "WIN_PCT" not in data.columns:
        raise KeyError("WIN_PCT")

    x = data[FEATURES]
    y = data["WIN_PCT"]
    return x, y


def train_winning_percentage_model(data, random_state=706):
    """Train a linear regression model on the eight Four Factors."""
    x, y = select_features(data)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }

    return model, metrics, predictions, y_test


def coefficient_report(model):
    """Compare each coefficient's sign with the basketball expectation."""
    rows = []
    for feature, slope in zip(FEATURES, model.coef_):
        expected_positive = EXPECTED_SIGN[feature] > 0
        matches = (slope > 0) == expected_positive
        direction = "positive" if expected_positive else "negative"
        rows.append(
            {
                "feature": feature,
                "coefficient": slope,
                "expected_direction": direction,
                "matches_basketball": matches,
            }
        )
    return pd.DataFrame(rows)


def predict_all_teams(data, model):
    """Add a predicted winning percentage for every team-season."""
    result = data.copy()
    x, _ = select_features(result)
    result["PREDICTED_WIN_PCT"] = model.predict(x)
    return result


def get_top_teams_by_year(data, model, number_of_teams=5):
    """Return the highest predicted teams in each season.

    If a season has fewer teams than requested, every team from that season
    is returned.
    """
    predicted_data = predict_all_teams(data, model)
    predicted_data = predicted_data.sort_values(
        ["YEAR", "PREDICTED_WIN_PCT"],
        ascending=[True, False],
    )

    top_teams = predicted_data.groupby("YEAR", group_keys=False).head(number_of_teams)
    top_teams = top_teams.copy()
    top_teams["RANK"] = top_teams.groupby("YEAR").cumcount() + 1
    return top_teams


def four_factor_means_by_year(data):
    """Yearly means for each of the eight Four Factors."""
    return data.groupby("YEAR")[FEATURES].mean()


def plot_actual_vs_predicted(actual, predicted, filepath="Figure_1.png"):
    """Save the actual-versus-predicted scatterplot and close the figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actual = pd.Series(actual).reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)

    plt.figure()
    plt.scatter(actual, predicted)
    plt.xlabel("Actual Winning Percentage")
    plt.ylabel("Predicted Winning Percentage")
    plt.title("Actual vs. Predicted Winning Percentage")
    plt.plot(
        [actual.min(), actual.max()],
        [actual.min(), actual.max()],
        color="red",
    )
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
