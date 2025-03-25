# Initial Experiments

A round of initial experiments must be performed to evaluate model performance.
The main idea is to train models with their default parameters to identify promising models.
There are some important questions to ask that the first rounds of experiments should be designed to answer.

## How Should Data Be Scaled?	
This is another important question.
There's more than one way to scale data, so this should be investigated as well.
The standard scaler works best for normally distributed variables, but EDA showed that variables were gamma distributed.
It comes down to whether the MinMax scaler or the robust scaler should be used.

## Should Statistical Outliers Be Removed?
As shown in the initial EDA, the data contains the presence of statistical outliers.
It's hard to deem them as actual outliers from a domain knowledge perspective, since those are valid cab rides.
If the statistical outliers are removed from the training data, it opens up the possibility that the selected model will perform poorly when statistical outliers are inputted for a prediction.
This would force the model to extrapolate and could go poorly.
Model performance should be evaluated on train and test datasets with and without outliers.

## How Much Data Is Good Enough?
One knob to turn is how much data should be used to train the model.
It's more than likely that a successful model would not need all 4 years of data to train.
There's already 3 months worth of data that has been pulled for the initial EDA.
By performing a train-test split with the test set receiving 20% of the data, that translates to training a model on about 2.4 months worth of data.
However, a better approach would be to do the following:
1. Train with 1 month of data and predict another month
2. Train with 2 months of data and predict another month
3. Train with 3 months of data and predict another month

For this experiment, June through August 2023 will be used as the 3 month time span for training and September 2023 will be used for testing.
This translates to:
1. Train with August 2023 and test with September 2023
2. Train with July and August 2023 and test with September 2023
3. Train with June July and August 2023 and test with September 2023

## Defining the First Set of Experiments
Obviously, we want to know if a model that can perform reasonably well can be trained.
So, the first set of experiments should focus on using a 3 month block of data for training (June-August 2023) and 1 month of data for testing (September 2023).
Also, the different scalers can be tested as well.
Let us start by trying to fit some models and also test to see how the presence of statistical outliers and different scalers make a difference.
The amount of training data can be addressed later.

## Models to Use
There are multiple ML models that will be used in the first round of experiments:
* LGBM with default parameters
* LGBM using Random Forest
* LGBM using DART
* XGBoost with default parameters
* CatBoost with default parameters
* Lasso using Lasso CV to find the best alpha, then retrain Lasso model with best alpha and evaluate
* Ridge using Ridge CV to find the best alpha, then retrain Ridge model with best alpha and evaluate
* ElasticNet using ElasticNet CV to find the best alpha, then retrain ElasticNet model with best alpha and l1 ratio and evaluate
* Linear Regression
* Huber Regression
* SGD
* Random Forest from Sklearn

Cross validation and validation metrics will be computed for each model.

Due to the data volume, SVM will be excluded for now.