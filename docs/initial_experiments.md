# Initial Experiments

Having performed EDA and data cleaning, the data is now ready for training and testing machine learning models.

## The Experiments
Let's use several models to fit the data and evaluate model performance.
The variable in this experiment will be the features used.
There will be 3 sets of features used to see which give the best result, and if the best result is good enough.
The feature sets are:
1. Trip duration and trip distance
2. Trip duration, trip distance, r, and $\theta$
3. R and $\theta$

Domain knowledge tells us that the trip duration and distance is how the fare is actually calculated.
Out of curiosity, I would personally like to know if polar coordinate representation of the trip duration and distance would make any difference in the performance.

## Models to Use
Several tree based algorithms will be used
* LightGBM GBT
* XGBoost
* CatBoost
* Random Forest (Scikit-learn)
* Random Forest (LighGBM)
* Light GBM Dart

Linear models will also be used:
* Linear regression
* Lasso
* Ridge
* Elastic Net
* Huber Regression
* Stochastic Gradient Descent

## Feature Scaling
Tree based models will have no feature scaling.
Linear models will use the MinMax scaler on all features.

