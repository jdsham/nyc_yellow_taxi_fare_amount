import xgboost as xgb
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from custom_funcs import plot_residuals, plot_true_vs_pred
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error
import argparse


def r2_xgb(y_pred:np.array, data:xgb.DMatrix) -> tuple:
    """A custom function to calculate the R-squared score for an XGBoost regression model

    Args:
        y_pred (np.array): the predicted values from the model
        data (xgb.DMatrix): the true values the model is trying to predict

    Returns:
        tuple: the name of the metric and the value
    """
    y_true = np.array(data.get_label())
    r2 = r2_score(y_true, y_pred)
    return "r2", r2


def rmse_xgb(y_pred:np.array, data:xgb.DMatrix) -> tuple:
    """A custom function to calculate the RMSE for an XGBoost regression model

    Args:
        y_pred (np.array): the predicted values from the model
        data (xgb.DMatrix): the true values the model is trying to predict

    Returns:
        tuple: the name of the metric and the value
    """
    y_true = np.array(data.get_label())
    rmse = root_mean_squared_error(y_true, y_pred)
    return "rmse", rmse


def mape_xgb(y_pred:np.array, data:xgb.DMatrix) -> tuple:
    """A custom function to calculate the MAPE for an XGBoost regression model

    Args:
        y_pred (np.array): the predicted values from the model
        data (xgb.DMatrix): the true values the model is trying to predict

    Returns:
        tuple: the name of the metric and the value
    """
    y_true = np.array(data.get_label())
    np.maximum(0.1, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return "mape", mape


def mae_xgb(y_pred:np.array, data:xgb.DMatrix) -> tuple:
    """A custom function to calculate the MAE for an XGBoost regression model

    Args:
        y_pred (np.array): the predicted values from the model
        data (xgb.DMatrix): the true values the model is trying to predict

    Returns:
        tuple: the name of the metric and the value
    """
    y_true = np.array(data.get_label())
    np.maximum(0.1, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return "mae", mae


def custom_metric_xgb(y_pred:np.array, data:xgb.DMatrix) -> list:
    """A custom function to calculate multiple evaluation metrics for an XGBoost regression model.
    Computes the R-squared, MAPE, RMSE, and MAE.

    Args:
        y_pred (np.array): the predicted values from the model
        data (xgb.DMatrix): the true values the model is trying to predict

    Returns:
        list: contains tuples of each score, where each tuple has the metric name and metric value
    """
    r2 = r2_xgb(y_pred, data)
    mape = mape_xgb(y_pred, data)
    mae = mae_xgb(y_pred, data)
    return [r2, mape, mae]


def run_xgboost(X_train:np.array, y_train:np.array, X_test:np.array, y_test:np.array, args:argparse.ArgumentParser, base_path:str, run_name:str, feature_names:list, metrics:dict, artifacts:list, W_train:np.array=None) -> tuple:
    """Uses the XGBoost gradient boosted trees regression model to train, perform CV, and evaluate model performance with validation data.
    This includes calculating metrics and generating plots to evaluate model performance.
    This function is called by a python file that runs the actual data science experiment.

    Args:
        X_train (np.array): Data to train the model
        y_train (np.array): Training data targets
        X_test (np.array): Data to test the model
        y_test (np.array): Testing data targets
        args (argparse.ArgumentParser): parsed Argparser arguments to specify the experiment run
        base_path (str): base path to save artifacts
        run_name (str): the name of the run of the experiment
        feature_names (list): names of the features
        metrics (dict): a dictionary containing computed metrics by name and value. These values are reported to MLFlow
        artifacts (list): A list of paths of each artifact (files) that was generated and saved. Artifacts are uploaded to MLFlow
        W_train (None | np.array): Training data weights if specified. Default is None.

    Returns:
        tuple: returns the metrics dictionary, artifacts list, and output parameter dictionary. These variables are reported to MLFlow
    """
    ####################
    # <Train the Model>
    ####################
    output_parameters = dict()
    model_name = args.model

    # Create XGBoost DMatrices
    train_data = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names, weight=W_train)
    valid_data = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    model_params = {"objective": "reg:squarederror",  "random_state": args.random_state,}
    # Add or update model parameters
    for key, val in args.model_params.items():
        model_params[key] = val

    base_metric_names = ["mae", "rmse", "r2", "mape"]
    results = xgb.cv(model_params, train_data, num_boost_round=10, nfold=args.cv, shuffle=True, metrics=["rmse"], custom_metric=custom_metric_xgb )
    results_mean = results.loc[float(args.cv-1)]
    
    evals = {}
    test_model = xgb.train(model_params, train_data, evals=[(valid_data,"valid_0"), (train_data, "training")], evals_result=evals,custom_metric=custom_metric_xgb)
    y_pred = test_model.predict(valid_data)
    #####################
    # </Train the Model>
    #####################

    ######################
    # <Calculate Metrics>
    ######################
    for metric_name in base_metric_names:
        metrics[f'cv_{metric_name}'] = results_mean[f"test-{metric_name}-mean"]

    metrics['test_mae'] = evals["valid_0"]["mae"][-1]
    metrics['test_rmse'] = evals["valid_0"]["rmse"][-1]
    metrics['test_r2'] = evals["valid_0"]["r2"][-1]
    metrics['test_mape'] = evals["valid_0"]["mape"][-1]
    #######################
    # </Calculate Metrics>
    #######################

    ##################
    # <Metric Curves>
    ##################
    # Errors / Residuals
    artifacts = plot_residuals(y_test, y_pred, model_name, run_name, base_path, artifacts)
    # Truth vs Prediction
    artifacts = plot_true_vs_pred(y_test, y_pred, model_name, run_name, base_path, artifacts)
    ###################
    # </Metric Curves>
    ###################

    #####################
    # <Evaluation Plots>
    #####################
    feat_importance_path = f"{base_path}/{model_name}_{run_name}_feature_importances.png"
    xgb.plot_importance(test_model)
    plt.savefig(feat_importance_path, bbox_inches='tight')
    artifacts.append(feat_importance_path)
    plt.close()

    for key in evals["training"].keys():
        metric_plot_path = f"{base_path}/{model_name}_{run_name}_learning_curve_{key}.png"
        lgb.plot_metric(evals, metric=key)
        plt.savefig(metric_plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        artifacts.append(metric_plot_path)
    ######################
    # </Evaluation Plots>
    ######################
    return metrics, artifacts, output_parameters