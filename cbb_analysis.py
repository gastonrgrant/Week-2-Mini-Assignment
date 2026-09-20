import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Select the Four Factors of Winning for our features
FEATURES = ["EFG_O", "TOR", "ORB", "FTR", "DRB", "TORD", "EFG_D", "FTRD"]


def load_data(filepath):
    # load cbb dataset
    return pd.read_csv(filepath)


def add_win_percentage(data):
    # Add win percentage column
    # Copy dataset so we do not alter our original
    result = data.copy()
    result["WIN_PCT"] = result["W"] / result["G"]
    return result


def select_features(data):
    # Select features/target for training and testing
    x = data[FEATURES]
    y = data["WIN_PCT"]
    return x, y


def train_winning_percentage_model(data, random_state=706):
    # Create and train a linear regression ML model to predict Winning Percentage
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


def predict_all_teams(data, model):
    #Add predicted winning percentages to the dataset for all teams
    result = data.copy()
    x, _ = select_features(result)
    result["PREDICTED_WIN_PCT"] = model.predict(x)
    return result


def get_top_teams_by_year(data, model, number_of_teams=5):
    # Top Predicted Teams for Each Year
    predicted_data = predict_all_teams(data, model)

    predicted_data = predicted_data.sort_values(
        ["YEAR", "PREDICTED_WIN_PCT"],
        ascending=[True, False],
    )

    top_teams = (
        predicted_data.groupby("YEAR")
        .head(number_of_teams)
        .copy()
    )

    top_teams["RANK"] = top_teams.groupby("YEAR").cumcount() + 1

    return top_teams