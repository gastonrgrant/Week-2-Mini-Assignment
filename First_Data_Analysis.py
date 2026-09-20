import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import polars as pl
import time

# Load the datasets in pandas and polars
cbb = pd.read_csv("archive/cbb.csv")
cbb_polars = pl.read_csv("archive/cbb.csv")

# Inspect the data set

print(f"\nCollege Basketball Data Set:\n {cbb.head(50)}\n")


cbb.info()

print(f"\nDescription of the dataset:\n{cbb.describe()}")

# check missing values in each column
missing_values = cbb.isnull().sum()
print(f"\nMissing values in each column:\n{missing_values}")

# check missing values in whole data set
missing_values_total = cbb.isnull().sum().sum()
print(f"\nTotal missing values in the dataset: {missing_values_total}")

# check for duplicates
duplicates = cbb.duplicated().sum()
print(f"\nNumber of duplicate rows in the dataset: {duplicates}")


# Data filtering

# Twenty win teams

twenty_win_teams = cbb[cbb["W"] >= 20]
print(f"\nTwenty-win teams:\n{twenty_win_teams.head(50)}")

# Tournament Teams

tournament_teams = cbb[cbb["POSTSEASON"].notnull()]

print(f"Tournament teams:\n{tournament_teams.head(50)}")

# Summary statistics of 3 Point Shooting Percentage (3P%) over the entire dateset.
print(f"Average 3 Point Shooting Percentage across all years:\n{cbb['3P_O'].mean()}")
print(f"\nMedian 3 Point Shooting Percentage across all years:\n{cbb['3P_O'].median()}")
print(
    f"\nRange of 3 Point Shooting Percentage across all years:\n{cbb['3P_O'].max() - cbb['3P_O'].min()}"
)
# Analysis of Dean Oliver's Four Factors of Basketball
# These are the four factors (both on offense and defense) that Dean Oliver identified as the most important for winning basketball games: Effective Field Goal Percentage (EFG), Turnover Rate (TOV), Offensive/Defensive Rebounds, and Free Throw Rate (FT).


# Average Effective Field Goal Percentage by Year (Offensive and Defensive)
print(
    f'\nAverage Effective Field Goal Percentage by Year (Offensive):\n {cbb.groupby("YEAR")["EFG_O"].mean()}'
)
print(
    f'\nAverage Effective Field Goal Percentage by Year (Defensive):\n {cbb.groupby("YEAR")["EFG_D"].mean()}'
)

# Average Turnover Rate by Year (Offensive and Defensive)
print(
    f'\nAverage Turnover Rate by Year (Offensive -Lower is better):\n {cbb.groupby("YEAR")["TOR"].mean()}'
)
print(
    f'\nAverage Turnover Rate by Year (Defensive - Higher is better):\n {cbb.groupby("YEAR")["TORD"].mean()}'
)

# Average Total Offensive and Defensive Rebounds by Year
print(
    f'\nAverage Total Offensive Rebounds by Year:\n {cbb.groupby("YEAR")["ORB"].mean()}'
)
print(
    f'\nAverage Total Defensive Rebounds by Year:\n {cbb.groupby("YEAR")["DRB"].mean()}'
)

# Average Free Throw Rate by Year (Offensive and Defensive)
print(
    f'\nAverage Free Throw Rate by Year (Offensive):\n {cbb.groupby("YEAR")["FTR"].mean()}'
)
print(
    f'\nAverage Free Throw Rate Allowed by Year (Defensive):\n {cbb.groupby("YEAR")["FTRD"].mean()}'
)


# ML Algorithm: Linear Regression to predict winning percentage based on Dean Oliver's Four Factors of Basketball

# Make x values the four factors and y values the winning percentage

x = cbb[["EFG_O", "TOR", "ORB", "FTR", "DRB", "TORD", "EFG_D", "FTRD"]]

# Create a new column for winning percentage
cbb["WIN_PCT"] = cbb["W"] / (cbb["G"])
y = cbb["WIN_PCT"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=706
)

# choose ML algorithm

ml_model = LinearRegression()

# Train the model
ml_model.fit(x_train, y_train)

# Predict the winning percentage for the test set

y_prediction = ml_model.predict(x_test)

# Evaluate

print(f"\nSlopes (m):")
for stat, slope in zip(x.columns, ml_model.coef_):
    print(f"{stat}: {slope}")
print(f"\nY Intercept (b): {ml_model.intercept_}")
print(f"\nMean Absolute Error: {mean_absolute_error(y_test, y_prediction)}")
print(f"\nMean Squared Error: {mean_squared_error(y_test, y_prediction)}")
print(f"\nR-squared: {r2_score(y_test, y_prediction)}\n")


# Plot actual versus predicted winning percentages
plt.scatter(y_test, y_prediction)

plt.xlabel("Actual Winning Percentage")
plt.ylabel("Predicted Winning Percentage")
plt.title("Actual vs. Predicted Winning Percentage")

# Add a perfect-prediction reference line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")

plt.show()


# Top Predicted Teams for Each Year

cbb["PREDICTED_WIN_PCT"] = ml_model.predict(x)

top_teams = (
    cbb.sort_values(
        ["YEAR", "PREDICTED_WIN_PCT"],
        ascending=[True, False]
    )
    .groupby("YEAR")
    .head(5)
)

top_teams["RANK"] = (
    top_teams.groupby("YEAR").cumcount() + 1
)

for year, group in top_teams.groupby("YEAR"):
    print(f"\n{year}")
    print(
        group[
            ["RANK", "TEAM", "PREDICTED_WIN_PCT"]
        ].to_string(index=False)
    )

# Polars Performance Comparison

print("\nPolars DataFrame Head:")
print(cbb_polars.head())

# Filter teams with 20 wins in Polars
polars_twenty_win_teams = cbb_polars.filter(pl.col("W") >= 20)

print("\nPolars Twenty-win teams:")
print(polars_twenty_win_teams.head())

# Calculate average 3 Point Shooting Percentage in Polars

polars_avg_3p_pct_allyears = cbb_polars.select(pl.col("3P_O").mean().alias("avg_3P_O"))
print(
    f"\nPolars Average 3 Point Shooting Percentage across all years:\n{polars_avg_3p_pct_allyears}"
)

# Calculate average Offensive Effective Field Goal Percentage by Year in Polars
polars_avg_efg_o_by_year = (
    cbb_polars.group_by("YEAR")
    .agg(pl.col("EFG_O").mean().alias("avg_EFG_O"))
    .sort("YEAR")
)
print(
    f"\nPolars Average Effective Field Goal Percentage by Year (Offensive):\n{polars_avg_efg_o_by_year}"
)

# Calculate the Pandas runtime

pandas_start = time.perf_counter()

pandas_avg_efg_o_by_year = cbb.groupby("YEAR")["EFG_O"].mean()

pandas_runtime = time.perf_counter() - pandas_start
print(f"Pandas runtime: {pandas_runtime}")
# Calculate Polars runtime

polars_start = time.perf_counter()
polars_avg_efg_o_by_year = (
    cbb_polars.group_by("YEAR")
    .agg(pl.col("EFG_O").mean().alias("avg_EFG_O"))
    .sort("YEAR")
)
polars_runtime = time.perf_counter() - polars_start
print(f"Polars runtime: {polars_runtime}")
