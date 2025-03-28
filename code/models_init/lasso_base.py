from sklearn.linear_model import LassoCV, Lasso
import numpy as np
from sklearn.model_selection import cross_validate
from custom_funcs import calc_cv_metrics_sklearn, calc_metrics_sklearn, plot_residuals, plot_true_vs_pred, plot_learning_curve
import argparse


def run_lasso(X_train:np.array, y_train:np.array, X_test:np.array, y_test:np.array, args:argparse.ArgumentParser, base_path:str, run_name:str, feature_names:list, metrics:dict, artifacts:list, W_train:np.array=None) -> tuple:
    """Uses the SKlearn Lasso and LassoCV models to train, perform CV, and evaluate model performance with validation data.
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
        tuple: returns the model, metrics dictionary, artifacts list, and output parameter dictionary. The model and variables are logged to MLFlow
    """
    ####################
    # <Train the Model>
    ####################
    output_parameters = dict()
    model_name = args.model

    model = LassoCV(cv=args.cv, random_state=args.random_state)
    model.fit(X_train, y_train, sample_weight=W_train)
    best_alpha = model.alpha_
    best_coefs = list(zip(feature_names,model.coef_))
    output_parameters["best_alpha"] = best_alpha
    output_parameters["best_coefs"] = best_coefs

    model_params = {"alpha":best_alpha, "random_state":args.random_state}
    # Add or update model parameters
    for key, val in args.model_params.items():
        model_params[key] = val

    model = Lasso(**model_params)
    scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "neg_mean_absolute_percentage_error", "r2"]
    cv = cross_validate(model, X_train, y_train, scoring=scoring, cv=args.cv, n_jobs=-1, params={"sample_weight": W_train})

    model = Lasso(alpha=best_alpha, random_state=args.random_state)
    model.fit(X_train, y_train, sample_weight=W_train)
    y_pred = model.predict(X_test)
    #####################
    # </Train the Model>
    #####################

    ######################
    # <Calculate Metrics>
    ######################
    metrics = calc_cv_metrics_sklearn(metrics, cv)
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
    artifacts = plot_learning_curve(Lasso(**model_params), X_train, y_train, model_name, run_name, artifacts, metric="rmse")
    ###################
    # </Metric Curves>
    ###################
    return model, metrics, artifacts, output_parameters