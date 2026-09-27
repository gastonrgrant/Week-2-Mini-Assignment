import time

import polars as pl

from cbb_analysis import (
    add_win_percentage,
    clean_team_data,
    coefficient_report,
    four_factor_means_by_year,
    get_top_teams_by_year,
    load_data,
    plot_actual_vs_predicted,
    train_winning_percentage_model,
)

DATA_PATH = "archive/cbb.csv"


def main():
    cbb = load_data(DATA_PATH)
    cbb_polars = pl.read_csv(DATA_PATH)

    print(f"\nCollege Basketball Data Set:\n {cbb.head(50)}\n")
    cbb.info()
    print(f"\nDescription of the dataset:\n{cbb.describe()}")

    missing_values = cbb.isnull().sum()
    print(f"\nMissing values in each column:\n{missing_values}")
    print(f"\nTotal missing values in the dataset: {missing_values.sum()}")
    print(f"\nNumber of duplicate rows in the dataset: {cbb.duplicated().sum()}")

    twenty_win_teams = cbb[cbb["W"] >= 20]
    print(f"\nTwenty-win teams:\n{twenty_win_teams.head(50)}")

    tournament_teams = cbb[cbb["POSTSEASON"].notnull()]
    print(f"Tournament teams:\n{tournament_teams.head(50)}")

    print(
        f"Average 3 Point Shooting Percentage across all years:\n{cbb['3P_O'].mean()}"
    )
    print(
        "\nMedian 3 Point Shooting Percentage across all years:\n"
        f"{cbb['3P_O'].median()}"
    )
    three_point_range = cbb["3P_O"].max() - cbb["3P_O"].min()
    print(
        f"\nRange of 3 Point Shooting Percentage across all years:\n{three_point_range}"
    )

    yearly_means = four_factor_means_by_year(cbb)
    print(
        "\nAverage Effective Field Goal Percentage by Year (Offensive):\n"
        f"{yearly_means['EFG_O']}"
    )
    print(
        "\nAverage Effective Field Goal Percentage by Year (Defensive):\n"
        f"{yearly_means['EFG_D']}"
    )
    print(
        "\nAverage Turnover Rate by Year (Offensive - Lower is better):\n"
        f"{yearly_means['TOR']}"
    )
    print(
        "\nAverage Turnover Rate by Year (Defensive - Higher is better):\n"
        f"{yearly_means['TORD']}"
    )
    print(f"\nAverage Offensive Rebounding Rate by Year:\n{yearly_means['ORB']}")
    print(f"\nAverage Defensive Rebounding Rate by Year:\n{yearly_means['DRB']}")
    print(f"\nAverage Free Throw Rate by Year (Offensive):\n{yearly_means['FTR']}")
    print(
        "\nAverage Free Throw Rate Allowed by Year (Defensive):\n"
        f"{yearly_means['FTRD']}"
    )

    modeling_data = clean_team_data(cbb)
    removed_rows = len(cbb) - len(modeling_data)
    print(f"\nRows removed before modeling: {removed_rows}")
    modeling_data = add_win_percentage(modeling_data)

    model, metrics, predictions, actual = train_winning_percentage_model(modeling_data)
    report = coefficient_report(model)

    print("\nSlopes (m):")
    print(report.to_string(index=False))
    print(f"\nY Intercept (b): {model.intercept_}")
    print(f"\nMean Absolute Error: {metrics['mae']}")
    print(f"\nMean Squared Error: {metrics['mse']}")
    print(f"\nR-squared: {metrics['r2']}\n")

    plot_actual_vs_predicted(actual, predictions)

    top_teams = get_top_teams_by_year(modeling_data, model)
    for year, group in top_teams.groupby("YEAR"):
        print(f"\n{year}")
        print(group[["RANK", "TEAM", "PREDICTED_WIN_PCT"]].to_string(index=False))

    print("\nPolars DataFrame Head:")
    print(cbb_polars.head())

    polars_twenty_win_teams = cbb_polars.filter(pl.col("W") >= 20)
    print("\nPolars Twenty-win teams:")
    print(polars_twenty_win_teams.head())

    polars_avg_3p = cbb_polars.select(pl.col("3P_O").mean().alias("avg_3P_O"))
    print(
        "\nPolars Average 3 Point Shooting Percentage across all years:\n"
        f"{polars_avg_3p}"
    )

    polars_avg_efg_o_by_year = (
        cbb_polars.group_by("YEAR")
        .agg(pl.col("EFG_O").mean().alias("avg_EFG_O"))
        .sort("YEAR")
    )
    print(
        "\nPolars Average Effective Field Goal Percentage by Year (Offensive):\n"
        f"{polars_avg_efg_o_by_year}"
    )

    pandas_start = time.perf_counter()
    cbb.groupby("YEAR")["EFG_O"].mean()
    pandas_runtime = time.perf_counter() - pandas_start
    print(f"Pandas runtime: {pandas_runtime}")

    polars_start = time.perf_counter()
    cbb_polars.group_by("YEAR").agg(pl.col("EFG_O").mean().alias("avg_EFG_O")).sort(
        "YEAR"
    )
    polars_runtime = time.perf_counter() - polars_start
    print(f"Polars runtime: {polars_runtime}")


if __name__ == "__main__":
    main()
