from sklearn.svm import LinearSVR
import numpy as np
from sklearn.model_selection import cross_validate
from custom_funcs import calc_cv_metrics_sklearn, calc_metrics_sklearn, plot_residuals, plot_true_vs_pred, plot_residual_descriptive_stats, plot_errors_to_features
import argparse


def run_ensemble(model, X_train:np.array, y_train:np.array, X_test:np.array, y_test:np.array, args:argparse.ArgumentParser, base_path:str, run_name:str, feature_names:list, metrics:dict, artifacts:list, W_train:np.array=None) -> tuple:
    """Uses Sklearn's Linear SVR model to train, perform CV, and evaluate model performance with validation data.
    This includes calculating metrics and generating plots to evaluate model performance.
    This function is called by a python file that runs the actual data science experiment.

    Args:
        model (Sci-kit Learn Model): The ensemble model to train
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
        tuple: returns the model, metrics dictionary, and artifacts list. The model and variables are logged to MLFlow
    """
    ####################
    # <Train the Model>
    ####################
    model_name = args.ensemble_type

    
    #scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "neg_mean_absolute_percentage_error", "r2"]
    #cv = cross_validate(model, X_train, y_train, scoring=scoring, cv=args.cv, n_jobs=4, params={"sample_weight": W_train})

    model.fit(X_train, y_train, sample_weight=W_train)
    y_pred = model.predict(X_test)
    #####################
    # </Train the Model>
    #####################

    ######################
    # <Calculate Metrics>
    ######################
    #metrics = calc_cv_metrics_sklearn(metrics, cv)
    metrics = calc_metrics_sklearn(metrics, y_test, y_pred)
    #######################
    # </Calculate Metrics>
    #######################

    ##################
    # <Metric Curves>
    ##################
    # Errors / Residuals
    artifacts = plot_residuals(y_test, y_pred, model_name, run_name, base_path, artifacts)
    artifacts = plot_residual_descriptive_stats(y_test, y_pred, model_name, run_name, base_path, artifacts)
    artifacts = plot_errors_to_features(X_test, y_test, y_pred, feature_names, model_name, run_name, base_path, artifacts)
    # Truth vs Prediction
    artifacts = plot_true_vs_pred(y_test, y_pred, model_name, run_name, base_path, artifacts)
    ###################
    # </Metric Curves>
    ###################
    return model, metrics, artifacts