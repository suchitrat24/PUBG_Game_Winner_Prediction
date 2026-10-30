# PUBG Game Winner Prediction

## Project Overview

This project analyzes PUBG match data and predicts each player's final placement (`winPlacePerc`, where 1 = winner and 0 = last place) from their in-game stats, using data science and machine learning techniques.

The project was completed as part of the **DataMites AI Engineer Program (Internship Project)**.

The primary objective was to perform exploratory data analysis on **4,446,966 player records (29 columns)** from **47,965 matches**, identify the in-game habits that go with a higher finish, build a regression model that predicts a player's placement, and give players practical suggestions on how to finish higher and win more matches.

## Project Objectives

- Perform data cleaning and exploratory data analysis (EDA).
- Handle missing values, "no value" codes, custom event modes, and likely cheaters.
- Create match-relative features so that matches of different sizes and lengths can be compared.
- Prevent data leakage between teammates by splitting the data by match.
- Build and compare multiple regression models.
- Tune hyperparameters and evaluate advanced ensembles.
- Provide data-driven suggestions to players.

## Technologies & Libraries

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

## Project Workflow

### 1. Data Cleaning and Preprocessing
- Examined dataset structure, data types, missing values, and duplicates (no duplicate rows, every `Id` unique).
- Dropped the **1 row** with a missing `winPlacePerc` (a solo match with only 1 player, so no placement exists).
- Removed the **35,267 rows** from custom / event modes (`normal-*`, `crash*`, `flare*`) and kept the 6 ranked modes, which hold 99.2% of the rows.
- Removed **357 rows** that looked like cheating (kills without moving, more than 30 kills, more than 10 road kills, kills from 1 km or more, more than 80 weapons, more than 40 heals).
- Found "no value" codes: `rankPoints = -1` in 38.3% of rows and `killPoints` / `winPoints = 0` in 59.7%.
- Created new features: `MATCH TYPE`, `PLAYERS IN MATCH`, `TEAM SIZE`, `TOTAL DISTANCE`, `HEALS AND BOOSTS`, `HEADSHOT RATE`, `KILLPLACE PERC`, and `WALK RANK` (the player's walking distance rank inside their own match).
- Converted `matchType` using Label Encoding and scaled features with `StandardScaler` for Linear Regression and KNN.

After cleaning, **4,411,341 rows** remained (35,625 removed), with no missing values and no duplicates.

### 2. Exploratory Data Analysis (EDA)
- Analyzed the distributions of placement, kills, damage, walking and ride distance, boosts, heals, and weapons picked up.
- Examined median placement by walking distance, weapons picked up, vehicle use, kill streaks, and revives.
- Studied how placement changes with each additional kill, boost, and heal.
- Used histograms, count plots, relational plots, pair plots, boxplots, line plots, bar charts, and correlation heatmaps.

### 3. Outlier Analysis
Used the Interquartile Range (IQR) method and boxplots to identify extreme values.

Key findings:
- The IQR method does not fit this data well. Most players have 0 kills, 0 heals, or 0 ride distance, so **11.5% of players** were flagged for kills and **24.9%** for ride distance (every player who used a vehicle).
- The top 8 players by kills (23-26 kills) all walked 2-4 km, picked up a normal number of weapons, and almost all won the match. They are real, very good players.
- All of these values were kept. The only extreme values removed were the clear cheating cases in the cleaning step.

### 4. Feature Selection
- `WALK RANK` had the strongest correlation with placement (**0.89**), higher than `walkDistance` (0.81), because it compares a player with others in the same match.
- `killPlace` (-0.72) and `KILLPLACE PERC` (-0.73) were strongly negative, and both were kept.
- **Removed `maxPlace`** because it is almost identical to `numGroups` (0.998).
- **Removed `killPoints`, `winPoints`, and `rankPoints`** because they have almost no correlation with placement (0.01) and are filled with "no value" codes in 38-60% of rows.
- Kept `groupId` and `matchId` aside (not as features) for splitting the data and averaging predictions per team.
- Final model used **28 features**.

### 5. Machine Learning Model Development

This project uses supervised regression to predict a player's final placement (`winPlacePerc`).

The following algorithms were implemented and evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. K-Nearest Neighbors (KNN) Regressor
4. Random Forest Regressor
5. Gradient Boosting Regressor

To keep training time manageable, models were trained on a random sample of **600 whole matches (56,168 players)**. The data was split **by match** (80/20, `GroupShuffleSplit`) into 44,883 training and 11,285 test players, so teammates are never split between training and testing (0 matches appear in both sets).

### 6. Model Evaluation and Comparison

Models were evaluated using:
- R²
- MAE and RMSE
- Share of players predicted within 0.1 of their real placement
- 5-fold cross-validation by match (`GroupKFold`)

| Model | Train R² | Test R² | MAE | RMSE | % within 0.1 |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.983 | **0.931** | **0.058** | **0.081** | **82.4%** |
| Gradient Boosting | 0.924 | 0.921 | 0.062 | 0.087 | 80.3% |
| Decision Tree | 0.946 | 0.909 | 0.066 | 0.093 | 77.9% |
| KNN | 0.911 | 0.893 | 0.073 | 0.100 | 73.8% |
| Linear Regression | 0.869 | 0.867 | 0.081 | 0.112 | 70.6% |

**5-fold cross-validation by match**

| Model | CV R² (mean) | CV R² (std) | CV MAE |
|---|---:|---:|---:|
| Gradient Boosting (tuned) | **0.936** | 0.002 | **0.056** |
| Random Forest | 0.932 | 0.002 | 0.057 |
| Gradient Boosting | 0.921 | 0.002 | 0.061 |
| Decision Tree | 0.910 | 0.002 | 0.065 |
| KNN | 0.889 | 0.003 | 0.073 |
| Linear Regression | 0.869 | 0.005 | 0.081 |

Linear Regression did unusually well (0.87) because the strongest features (`WALK RANK`, `killPlace`) have an almost straight-line relationship with placement.

### 7. Hyperparameter Tuning and Advanced Ensembles
- **GridSearchCV** on Random Forest (`n_estimators` x `max_depth`) showed that depth matters and tree count does not. Depth 6 gave a CV R² of 0.899, and from depth 15 the score levelled off at about 0.930-0.931.
- **RandomizedSearchCV** on Random Forest (20 combinations) and HistGradientBoosting (30 combinations).
- **Voting** (average of tuned Random Forest, HistGradientBoosting, and Gradient Boosting) and **Stacking** (Random Forest, HistGradientBoosting, Extra Trees, and KNN with a Ridge meta-model) ensembles.
- **Team-averaged predictions:** all teammates share one placement, so each team's predictions were averaged and given to every member.

| Model | Test R² | Test MAE | % within 0.1 | CV R² (mean) |
|---|---:|---:|---:|---:|
| Stacking | 0.9368 | 0.0561 | 83.4% | **0.9392** |
| Voting | 0.9366 | 0.0562 | 83.4% | 0.9387 |
| HistGradientBoosting (random search) | 0.9358 | 0.0568 | 83.1% | 0.9387 |
| Random Forest (random search) | 0.9316 | 0.0576 | 82.8% | 0.9326 |
| Random Forest (grid search) | 0.9312 | 0.0580 | 82.6% | 0.9324 |
| Random Forest (hand-picked) | 0.9311 | 0.0580 | 82.4% | 0.9323 |
| **Voting + team average** | **0.9565** | **0.0483** | **88.4%** | - |

Tuning Random Forest changed nothing. Boosting and the ensembles improved CV R² by about **0.006**, about 3 times the variation between folds. Averaging predictions per team gave the biggest gain of all, raising the test R² from **0.937 to 0.957**.

**Best Performing Model: Voting Regressor with team-averaged predictions**

The Voting model ties for the best cross-validated R² with Stacking and HistGradientBoosting (differences of about 0.0005, well within the fold-to-fold variation). It was chosen because it is simpler than stacking (a plain average with no meta-model) and less dependent on one model's mistakes than HistGradientBoosting alone.

### 8. Final Model Performance

| Metric | Result |
|---|---:|
| Test R² | 0.9565 |
| Test MAE | 0.0483 |
| Test RMSE | 0.064 |
| Predictions within 0.05 | 61.9% |
| Predictions within 0.1 | 88.4% |
| Predictions within 0.2 | 99.5% |

- Mean error by mode: **0.041 in duo**, **0.047 in solo**, and **0.053 in squad**.
- Of the players predicted to finish in the top 10%, **77.9%** really did, and the model found **48.0%** of all real top-10% players.
- Permutation importance showed `KILLPLACE PERC`, `kills`, and `WALK RANK` as the most important features.

## Key Findings

- Walking distance had the clearest pattern: players who walked under 100 m had a median placement of **0.074**, compared with about **0.90** for 3 km or more.
- Among players who walked the most in their match (top 10%), about **half** finished in the top 10% (52.9% duo, 53.2% solo, 49.4% squad).
- Even at the same walking distance, more boosts went with a higher placement (solo 1-2 km: **0.660** with 0 boosts vs **0.947** with 7+).
- Kills matter most in solo and least in squad: the top-10% chance with 8+ kills was **86.6%** in solo vs **69.2%** in squad.
- Vehicles looked much better overall (0.729 vs 0.340), but at the same walking distance they only helped players who walked little. Among long survivors (3 km+), players without a vehicle placed slightly higher.
- Reviving at least one teammate raised the top-10% chance from **8%** to **23% (duo) / 21% (squad)**.
- Placement stopped improving after about 5 weapons and about 3 heals.

## Player Recommendations

- Keep moving with the safe zone and avoid camping in one spot.
- Pick up and use boosts regularly, especially before fights and zone moves.
- Take fights you can win, which matters most in solo.
- Use vehicles to move across the map early, but leave them before the final zones.
- Revive and support teammates in duo and squad.
- Don't over-loot: get a main weapon, some boosts and heals, then focus on moving and positioning.
- Compare your stats with the model's predicted placement to review your game.

## Challenges and Solutions

Some of the main challenges encountered included:

- **Very large dataset:** Used the full data for analysis and a sample of 600 whole matches for modelling.
- **Custom event modes and cheaters:** Removed event-mode rows and rows with impossible stats.
- **Unreliable columns:** Dropped `rankPoints`, `killPoints`, and `winPoints` because of their mixed "no value" codes.
- **Stats depend on the match:** Created `WALK RANK` and `KILLPLACE PERC` to compare players within the same match.
- **Zero-heavy, skewed features:** Kept real extreme values instead of removing IQR "outliers".
- **Teammates share one placement:** Averaged predictions per team.
- **Misleading group comparisons:** Compared players within the same match type and walking distance before making suggestions.

## Disclaimer

This project was completed for educational and portfolio purposes as part of the DataMites AI Engineer Program. The model results are based on historical PUBG match data and should not be interpreted as a validated production system.

## Author

**Suchitra Thanumalayan**

Data Science | Machine Learning | Artificial Intelligence
