# College Basketball Data Analysis

## Overview of Project

This project performs a beginner-level analysis of Division I college basketball data using Python, Pandas, Matplotlib, and scikit-learn. The dataset contains team-level statistics from the 2013–2019 and 2021–2025 seasons. The 2020 season is not included because the NCAA Tournament was canceled that year due to COVID-19.

The analysis includes importing and inspecting the data, checking for missing values and duplicates, filtering meaningful subsets, calculating summary statistics with `groupby()`, creating a visualization, and exploring a linear regression model.

## Research Question

Our machine learning analysis in this project focuses on answering the research question:
"How well do Dean Oliver's Four Factors of Basketball explain winning percentage in college basketball?"

Dean Oliver's four factors are effective field-goal percentage, turnover rate, rebounding rate, and free-throw rate. Each factor is measured on offense and on defense, so the model uses eight inputs. Defense is part of the Four Factors, not a separate question.

## Dataset

The dataset is from Kaggle and was scraped from Bart Torvik's college basketball rankings. It includes team performance, conference, efficiency, shooting, rebounding, turnover, free-throw, postseason, seed, and season information.

The dataset can be downloaded here: https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset?resource=download

Important variables include:

- `W` and `G`: wins and games played
- `ADJOE`: adjusted offensive efficiency
- `ADJDE`: adjusted defensive efficiency
- `EFG_O` and `EFG_D`: offensive and defensive effective field-goal percentage
- `TOR` and `TORD`: turnovers committed and turnovers generated
- `ORB` and `DRB`: offensive rebounding and defensive rebounding rates
- `FTR` and `FTRD`: offensive and defensive free-throw rates
- `3P_O`: offensive three-point shooting percentage
- `ADJ_T`: adjusted tempo
- `POSTSEASON` and `SEED`: tournament outcome and seed

## Analysis

### Data inspection

The script uses `head()` to display the first 50 rows, `info()` to inspect the columns and data types, and `describe()` to view numerical summary statistics. It also uses `isnull().sum()` to count missing values in each column and across the entire dataset, and `duplicated().sum()` to check for exact duplicate rows.

Missing values are expected in `POSTSEASON` and `SEED` because most teams do not reach the NCAA Tournament. In this file, both columns are missing for 3,433 of 4,249 team-seasons. These missing values are not automatically errors; they indicate that a team did not receive a postseason result or tournament seed.

There are no exact duplicate rows. Two team-seasons have more wins than games played: Southern Utah in 2021 (20 wins in 19 games) and McNeese State in 2024 (30 wins in 29 games). A winning percentage above 1 is not a real result, so those two rows are removed before the regression. Games played otherwise ranges from 5 to 40. Offensive three-point percentage ranges from about 24.7 to 44.1, which is a wide but believable college range, so those rows are kept.

### Filters

Two meaningful subsets are created:

1. `twenty_win_teams` contains teams with at least 20 wins. This identifies teams that had a relatively successful regular season.
2. `tournament_teams` contains teams with a nonmissing `POSTSEASON` value. This identifies teams that made the NCAA Tournament.

The cleaned dataset is used for the regression so that the model includes successful, average, and weaker teams rather than only a selected group.

### Summary statistics and grouping

The script calculates the overall mean, median, and range of offensive three-point percentage (`3P_O`).

Additionally, the script groups the data by `YEAR` and calculates yearly means for Dean Oliver's Four Factors on both ends of the floor (Dean Oliver is considered the father of basketball analytics, and his Four Factors are the four statistics he identified as the most impactful to winning basketball games in his 2004 book *Basketball on Paper*):

- effective field-goal percentage, offensive and defensive;
- turnover rate and turnovers generated;
- offensive and defensive rebounding;
- free-throw rate, offensive and defensive.

The offensive and defensive statistics are examined separately because the two sides of the game measure different aspects of performance. For example, lower `TOR` is generally better because it represents turnovers committed, while higher `TORD` is generally better because it represents turnovers generated from opponents.

## Machine Learning: Linear Regression

The model uses both sides of Dean Oliver's Four Factors as inputs:

```text
EFG_O, TOR, ORB, FTR, EFG_D, TORD, DRB, FTRD
```

The outcome is winning percentage, calculated as:

```text
WIN_PCT = W / G
```

Using winning percentage instead of raw wins makes comparisons fairer when teams play different numbers of games.

The data is divided into training and testing sets. Eighty percent of the rows are used to train the model, and 20% are reserved to test its predictions. `random_state=706` makes the random split reproducible, meaning the same rows are selected each time the script runs.

The model is evaluated with mean absolute error, mean squared error, and R-squared. The coefficients are also printed so that the estimated relationship for each Four Factor can be examined.

For the current run, after the two impossible win totals were removed, the model produced:

- Mean absolute error: approximately `0.059`, meaning predictions were off by about 5.9 percentage points on average.
- Mean squared error: approximately `0.0055`. Lower values indicate smaller prediction errors.
- R-squared: approximately `0.828`, meaning the eight Four Factors explained about 82.8% of the variation in winning percentage in the test data.

Each coefficient was also checked against the direction basketball would suggest. A higher `EFG_O`, `ORB`, `FTR`, `TORD`, or `DRB` should go with a higher winning percentage. A higher `TOR`, `EFG_D`, or `FTRD` should go with a lower one.

| Factor | Coefficient | Expected sign | Matches |
|---|---:|---|---|
| `EFG_O` | 0.0260 | positive | yes |
| `TOR` | -0.0225 | negative | yes |
| `ORB` | 0.0089 | positive | yes |
| `FTR` | 0.0032 | positive | yes |
| `DRB` | -0.0119 | positive | no |
| `TORD` | 0.0247 | positive | yes |
| `EFG_D` | -0.0231 | negative | yes |
| `FTRD` | -0.0038 | negative | yes |

Seven of the eight signs match that expectation. `DRB` is the exception: its coefficient is negative even though a higher defensive rebounding rate should help a team. That does not mean defensive rebounding causes losses. All eight factors are in the model at once, so each coefficient is the leftover association after the others are already accounted for. These results show a strong relationship between the Four Factors and winning percentage. They do not prove that any individual factor causes teams to win. The model also leaves out other influences, such as strength of schedule, injuries, and coaching.

## Visualization

![Actual vs. Predicted Winning Percentage](Figure_1.png)

The script saves a scatterplot, `Figure_1.png`, comparing the model's predicted winning percentages with the teams' actual winning percentages in the testing dataset. Saving the figure, instead of opening a window, lets the same script finish inside a Docker container.

The actual winning percentage appears on the x-axis, and the predicted winning percentage appears on the y-axis. Each point represents one team in the testing data.

The red diagonal line represents perfect predictions. Points on the red line would have predicted winning percentages exactly equal to their actual winning percentages. Points farther from the line represent larger prediction errors.

This visualization helps evaluate how closely the linear regression model's predictions match the actual outcomes. The overall pattern of the points shows that the model captures a meaningful relationship between the Four Factors and winning percentage, although the predictions are not perfect.

## How to Run

From the project folder, install the dependencies and run the script:

```bash
python -m pip install -r requirements.txt
python First_Data_Analysis.py
```

The script expects the dataset at:

```text
archive/cbb.csv
```

## Docker

The analysis can also run in a container. The Dockerfile starts from Python 3.11, installs `requirements.txt`, and copies the scripts plus `archive/cbb.csv`. Building the image packages that environment. Running the container starts the script, writes `Figure_1.png` inside the container, and exits when the script finishes.

```bash
docker build -t cbb-analysis .
docker run --rm cbb-analysis
```

`docker images` lists the built image. `docker ps` shows containers that are still running. This container exits when the analysis is done, so it will not stay in that list.

Add a resized screenshot of the image build and the running container here.

## Refactoring

`First_Data_Analysis.py` used to repeat the model training, the test-set predictions, and the top-teams ranking that `cbb_analysis.py` already defined. The script now calls those functions. Cleaning invalid win totals, the yearly Four Factor means, the scatterplot, and the coefficient sign check were also moved into `cbb_analysis.py` so the script is a short `main()` instead of one long block.

That change matters because the tests import `cbb_analysis.py`. If the script kept its own copy of the model, a later edit could change one path and leave the other behind while the tests still passed.

The project was checked by running `python -m pytest -q` and by running `python First_Data_Analysis.py`, which rewrote `Figure_1.png` and printed the same metrics described above. `black` formatted the Python files, and `flake8` reported no issues.

Add a screenshot of the GitHub commit diff for this refactor here.

## Polars Analysis

In addition to Pandas, this project uses Polars to repeat a data-grouping operation. Polars is another DataFrame library designed to process data efficiently, especially when working with larger datasets.

The same college basketball CSV file was loaded with Polars. Polars was used to group teams by `YEAR` and calculate the mean offensive effective field-goal percentage (`EFG_O`) for each season. This is equivalent to the Pandas grouping operation used earlier in the script. The Polars results were then sorted by `YEAR` so that the seasons appeared chronologically.

## Pandas and Polars Runtime Comparison

The runtime comparison measured how long Pandas and Polars took to group the data by `YEAR` and calculate the mean of `EFG_O`. Timing began immediately before each grouping operation and ended immediately afterward, so it measured the operations themselves rather than package installation or dataset loading.

In one measured run, the results were:

| Library | Runtime |
|---|---:|
| Pandas | 0.000243 seconds |
| Polars | 0.000588 seconds |

Pandas was slightly faster in this run. The difference was less than a millisecond. Because this dataset contains only about 4,249 rows and the operation is simple, both libraries completed the task almost immediately.

This result does not mean that Pandas is always faster. Polars is designed to provide larger performance benefits with larger datasets or more complicated operations. For this particular small dataset and operation, the runtime difference was too small to have practical significance.

## Testing and Reproducibility

This project includes automated tests using pytest. The tests validate the main components of the analysis:

- Loading the basketball dataset
- Calculating winning percentage
- Rejecting a row with zero games played
- Dropping a row with more wins than games
- Training and evaluating the linear regression model
- Raising an error when a Four Factor column is missing
- Generating the top predicted teams by year, including a request for more teams than a season contains

The project contains nine tests, including a system/integration test that checks the workflow from model training through top-team prediction.

To run the tests locally:

```bash
python -m pytest -q
```

GitHub Actions runs those tests, plus `black --check` and `flake8`, on Python 3.11 and 3.12.

[![Run Tests](https://github.com/gastonrgrant/Week-2-Mini-Assignment/actions/workflows/tests.yml/badge.svg)](https://github.com/gastonrgrant/Week-2-Mini-Assignment/actions/workflows/tests.yml)

<img width="1316" height="720" alt="Screenshot 2026-09-20 at 4 05 38 PM" src="https://github.com/user-attachments/assets/0c3f42ae-f255-4c50-b068-3e3e697d9620" />

## Files

- `First_Data_Analysis.py`: Main Python data analysis script
- `cbb_analysis.py`: Reusable functions used by the script and the tests
- `tests/test_cbb_analysis.py`: Unit and integration tests, including edge cases
- `.github/workflows/tests.yml`: GitHub Actions workflow
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container build for the analysis
- `.flake8`: Flake8 settings aligned with Black
- `.gitignore`: Files excluded from Git tracking
- `archive/cbb.csv`: College basketball dataset
- `Figure_1.png`: Actual versus predicted winning-percentage plot
- `README.md`: Project documentation
