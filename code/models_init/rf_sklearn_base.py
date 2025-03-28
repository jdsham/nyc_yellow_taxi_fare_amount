from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from custom_funcs import calc_metrics_sklearn, plot_residuals, plot_true_vs_pred, plot_learning_curve
import argparse

def run_rf_sklearn(X_train:np.array, y_train:np.array, X_test:np.array, y_test:np.array, args:argparse.ArgumentParser, base_path:str, run_name:str, feature_names:list, metrics:dict, artifacts:list, W_train:np.array=None) -> tuple:
    """Uses the SKlearn RandomForestRegressor model to train and evaluate model performance with validation data.
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

    model_params = {"max_depth":6, "n_jobs":-1, "random_state":args.random_state, "oob_score":True}
    # Add or update model parameters
    for key, val in args.model_params.items():
        model_params[key] = val

    # CV will be skipped and instead OOB scoring will be used to evaluate the model
    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #####################
    # </Train the Model>
    #####################

    ######################
    # <Calculate Metrics>
    ######################
    metrics = calc_metrics_sklearn(metrics, y_test, y_pred, data_type="cv")
    metrics = calc_metrics_sklearn(metrics, y_test, y_pred)
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
    # Learning Curve
    #artifacts = plot_learning_curve(RandomForestRegressor(max_depth=6, n_jobs=2, random_state=args.random_state), X_train, y_train, model_name, run_name, artifacts, metric="rmse")
    # Feature Importances
    feat_importance_path = f"{base_path}/{model_name}_{run_name}_feature_importances.png"
    feat_importances = pd.Series(model.feature_importances_, index=feature_names)
    feat_importances = feat_importances.sort_values(ascending=False)
    fig = plt.subplots(figsize=(10,10))
    feat_importances.plot(kind="barh", color="blue")
    plt.savefig(feat_importance_path, dpi=200, bbox_inches='tight')
    artifacts.append(feat_importance_path)
    plt.close()
    ###################
    # </Metric Curves>
    ###################

    return metrics, artifacts, output_parameters