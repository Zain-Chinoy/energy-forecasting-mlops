# Experiment results

Final model: **hist_gradient_boosting** on feature set **calendar_temp** (9 features).

## Final test-period metrics (evaluated once)

| test_rmse | test_mae | test_r2 | test_mape_pct | baseline_test_rmse | rmse_improvement_vs_baseline_pct | cv_rmse_mean | cv_rmse_std |
|---|---|---|---|---|---|---|---|
| 92.83 | 77.74 | 0.75 | 6.22 | 197.64 | 53.03 | 77.32 | 13.54 |

## Distribution-shift check (Jan-Jun 2020, not used for any decision)

| shift2020_rmse | shift2020_mae | shift2020_r2 | shift2020_mape_pct |
|---|---|---|---|
| 157.80 | 125.28 | 0.20 | 10.73 |

## Exploration: cross-validated results (training period)

| model_type | feature_set | n_features | cv_rmse_mean | cv_rmse_std | cv_mae_mean | cv_r2_mean |
|---|---|---|---|---|---|---|
| hist_gradient_boosting | calendar_temp | 9 | 77.45 | 13.59 | 65.87 | 0.82 |
| random_forest | calendar_only | 6 | 78.11 | 15.66 | 65.77 | 0.82 |
| random_forest | calendar_tocumen | 10 | 78.21 | 15.24 | 66.82 | 0.82 |
| hist_gradient_boosting | calendar_only | 6 | 78.21 | 15.44 | 65.54 | 0.82 |
| random_forest | calendar_temp | 9 | 78.23 | 13.55 | 66.70 | 0.82 |
| hist_gradient_boosting | calendar_tocumen | 10 | 78.39 | 14.77 | 66.98 | 0.82 |
| random_forest | all_features | 18 | 79.43 | 14.47 | 67.94 | 0.81 |
| random_forest | all_minus_wind | 15 | 79.51 | 14.86 | 68.14 | 0.81 |
| hist_gradient_boosting | all_minus_wind | 15 | 79.65 | 15.12 | 68.43 | 0.81 |
| random_forest | calendar_temp_humidity | 12 | 79.86 | 14.28 | 68.53 | 0.81 |
| hist_gradient_boosting | all_features | 18 | 80.00 | 14.46 | 68.68 | 0.81 |
| random_forest | all_minus_precip | 15 | 80.15 | 13.87 | 68.74 | 0.81 |
| hist_gradient_boosting | calendar_temp_humidity | 12 | 80.29 | 15.01 | 69.11 | 0.81 |
| hist_gradient_boosting | all_minus_precip | 15 | 81.17 | 14.32 | 69.74 | 0.80 |
| linear_regression | calendar_only | 6 | 96.58 | 15.12 | 77.65 | 0.72 |
| linear_regression | calendar_tocumen | 10 | 96.92 | 15.28 | 78.87 | 0.72 |
| linear_regression | all_minus_wind | 15 | 96.95 | 15.52 | 78.74 | 0.72 |
| linear_regression | calendar_temp | 9 | 97.17 | 15.30 | 78.51 | 0.72 |
| linear_regression | all_features | 18 | 97.37 | 15.60 | 79.19 | 0.72 |
| linear_regression | all_minus_precip | 15 | 98.14 | 15.55 | 79.84 | 0.72 |
| linear_regression | calendar_temp_humidity | 12 | 98.17 | 15.52 | 79.77 | 0.72 |
| random_forest | weather_only | 12 | 131.77 | 9.94 | 107.24 | 0.50 |
| hist_gradient_boosting | weather_only | 12 | 135.08 | 11.21 | 109.04 | 0.47 |
| linear_regression | weather_only | 12 | 142.24 | 9.30 | 116.34 | 0.41 |
| baseline_mean | none | 0 | 196.43 | 10.04 | 163.74 | -0.12 |

## Hyperparameter search

| params | cv_rmse_mean | cv_rmse_std | cv_mae_mean | cv_r2_mean |
|---|---|---|---|---|
| {"learning_rate": 0.03, "max_iter": 400, "max_depth": null, "min_samples_leaf": 20, "l2_regularization": 1.0} | 77.32 | 13.54 | 66.00 | 0.82 |
| {"learning_rate": 0.05, "max_iter": 200, "max_depth": null, "min_samples_leaf": 100, "l2_regularization": 0.1} | 77.32 | 13.44 | 65.90 | 0.82 |
| {"learning_rate": 0.03, "max_iter": 600, "max_depth": 10, "min_samples_leaf": 20, "l2_regularization": 0.1} | 77.35 | 13.64 | 65.83 | 0.82 |
| {"learning_rate": 0.1, "max_iter": 200, "max_depth": 10, "min_samples_leaf": 20, "l2_regularization": 0.1} | 77.40 | 13.64 | 65.80 | 0.82 |
| {"learning_rate": 0.03, "max_iter": 200, "max_depth": 10, "min_samples_leaf": 20, "l2_regularization": 0.1} | 77.45 | 13.39 | 66.16 | 0.82 |
| {"learning_rate": 0.03, "max_iter": 200, "max_depth": 6, "min_samples_leaf": 20, "l2_regularization": 0.0} | 77.50 | 13.35 | 66.20 | 0.82 |
| {"learning_rate": 0.03, "max_iter": 200, "max_depth": null, "min_samples_leaf": 20, "l2_regularization": 0.1} | 77.52 | 13.35 | 66.20 | 0.82 |
| {"learning_rate": 0.2, "max_iter": 600, "max_depth": null, "min_samples_leaf": 50, "l2_regularization": 0.1} | 78.03 | 13.66 | 66.00 | 0.82 |

Best parameters: `{"learning_rate": 0.03, "max_iter": 400, "max_depth": null, "min_samples_leaf": 20, "l2_regularization": 1.0}`
