# Data Preparation for ML Models

## Filtering Data

### Outliers to Remove
Consider removing outliers from the following columns:
1. Trip Distance
2. Trip Duration

This consideration is difficult to decide.
It is possible that what might be considered statistical outliers may not be actual outliers.
However, the presence of outliers can negatively impact model performance.
The concern is that a model trained without outliers must extrapolate when given outlier parameters (trip distance and duration) and thus model performance will suffer.
From a domain knowledge perspective, it is hard to say that valid trips with valid data are outliers at this point.
An experiment should be done to evaluate how the presence of statistical outliers impacts model performance.

## Transformations

### Features to add
Travel rates could be added, but the correlation is weak and domain knowledge suggests it is not relevant.
It is probably best to leave this feature out for the initial round of model fitting.

### Scaling Numerical Features
Features can be scaled for linear models in the following ways:
1. Standard scaling (assumes data is normally distributed, which isn't always valid)
2. MinMax scaling
3. Robust scaler

For tree models, features can simply be left as-is.