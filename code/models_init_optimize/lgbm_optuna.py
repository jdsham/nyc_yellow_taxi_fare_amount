import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from custom_funcs import plot_residuals, plot_true_vs_pred, plot_residual_descriptive_stats, plot_errors_to_features
import argparse
import optuna
plt.switch_backend('agg')


def r2_lgbm(y_pred:np.array, data:lgb.Dataset) -> tuple:
    """For use with LightGBM API (not compatible with Sklearn API).
    Computes the R^2 metric

    Args:
        y_pred (np.array): Predictions generated from LGBM Model
        data (Dataset): Dataset containing y_true data.

    Returns:
        tuple: metric name, the metric, if greater metric is better
    """
    y_true = np.array(data.get_label())
    r2 = r2_score(y_true, y_pred)
    return "r2", r2, False


def objective(trial, args, train_data):
    num_leaves = trial.suggest_int("num_leaves", 31,200, step=1)
    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 5, 50, step=5)
    num_iterations = trial.suggest_int("num_iterations", 100,500,step=50)
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, step=0.001)
    #min_gain_to_split = trial.suggest_float("min_gain_to_split", 0, 1, step=0.01)
    model_params = {"objective": "regression", "metric": "rmse", "random_state": args.random_state,
                    "feature_pre_filter":False,
                    "early_stopping_round": 30,
                    "num_leaves": num_leaves,
                    "min_data_in_leaf": min_data_in_leaf,
                    "num_iterations": num_iterations,
                    "learning_rate": learning_rate,
                    #"min_gain_to_split": min_gain_to_split,
                    "force_row_wise": True,
                    "first_metric_only": True}
    cv_results = lgb.cv(model_params, train_data, nfold=args.cv, shuffle=True, stratified=False)
    cv_rmse = cv_results["valid rmse-mean"][-1]
    return cv_rmse


def run_lgbm(X_train:np.array, y_train:np.array, X_test:np.array, y_test:np.array, args:argparse.ArgumentParser, base_path:str, run_name:str, feature_names:list, metrics:dict, artifacts:list, W_train:np.array=None) -> tuple:
    """Uses the LightGBM gradient boosted trees regression model to train, perform CV, and evaluate model performance with validation data.
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
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    model_params = {"objective": "regression", "metric": "l1,rmse,mape,r2,custom", "random_state": args.random_state}
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args, train_data), n_trials=30)
    trial = study.best_trial
    best_params = trial.params
    output_parameters = best_params

    model_params = model_params | best_params
    cv_results = lgb.cv(model_params, train_data, nfold=args.cv, shuffle=True, feval=[r2_lgbm], stratified=False)
    evals = {}
    model = lgb.train(model_params, train_data, valid_sets=[valid_data, train_data], feval=[r2_lgbm], callbacks = [lgb.record_evaluation(evals)])
    y_pred = model.predict(X_test)
    #####################
    # </Train the Model>
    #####################

    ######################
    # <Calculate Metrics>
    ######################
    metrics['cv_mae'] = cv_results["valid l1-mean"][-1]
    metrics['cv_rmse'] = cv_results["valid rmse-mean"][-1]
    metrics['cv_mape'] = cv_results["valid mape-mean"][-1]
    metrics['cv_r2'] = cv_results["valid r2-mean"][-1]

    metrics['test_mae'] = evals["valid_0"]["l1"][-1]
    metrics['test_rmse'] = evals["valid_0"]["rmse"][-1]
    metrics['test_mape'] = evals["valid_0"]["mape"][-1]
    metrics['test_r2'] = evals["valid_0"]["r2"][-1]
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

    #####################
    # <Evaluation Plots>
    #####################
    feat_importance_path = f"{base_path}/{model_name}_{run_name}_feature_importances.png"
    lgb.plot_importance(model)
    plt.savefig(feat_importance_path, bbox_inches='tight')
    artifacts.append(feat_importance_path)
    plt.close()

    for key in evals["training"].keys():
        metric_plot_path = f"{base_path}/{model_name}_{run_name}_learning_curve_{key}.png"
        lgb.plot_metric(evals, metric=key)
        plt.savefig(metric_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        artifacts.append(metric_plot_path)
    ######################
    # </Evaluation Plots>
    ######################
    return model, metrics, artifacts, output_parameters