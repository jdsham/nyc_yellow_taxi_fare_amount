import catboost as ctb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from custom_funcs import plot_residuals, plot_true_vs_pred, plot_residual_descriptive_stats, plot_errors_to_features
import argparse
from numpy.typing import ArrayLike
plt.switch_backend('agg')


def plot_feature_importance(importance:ArrayLike, names:list) -> None:
    """Plots the feature importances generated from a fitted Catboost model

    Args:
        importance (ArrayLike): feature importances of the fitted model
        names (list): feature names
    """
    model_type = "CatBoost"
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    # Create a DataFrame using a dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    # Define size of bar plot
    plt.figure(figsize=(10,8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + 'Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')



def run_catboost(X_train:np.array, y_train:np.array, X_test:np.array, y_test:np.array, args:argparse.ArgumentParser, base_path:str, run_name:str, feature_names:list, metrics:dict, artifacts:list, W_train:np.array=None) -> tuple:
    """Uses the CatBoost regression model to train, perform CV, and evaluate model performance with validation data.
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

    # Create Catboost Pools
    train_data = ctb.Pool(X_train, label=y_train, feature_names=feature_names, weight=W_train)
    valid_data = ctb.Pool(X_test, label=y_test, feature_names=feature_names)

    model_params = {"objective": "RMSE", "num_boost_round":100, "eval_metric": "RMSE", "custom_metric": ["MAE", "R2", 'MAPE'], "verbose":False,"random_state": args.random_state,}
    # Add or update model parameters
    for key, val in args.model_params.items():
        model_params[key] = val

    results = ctb.cv(params=model_params, pool=train_data, nfold=args.cv, shuffle=True)
    results_mean = results.loc[results.shape[0]-1]

    model = ctb.train(pool=train_data, params=model_params, eval_set=[valid_data])
    y_pred = model.predict(X_test)
    #####################
    # </Train the Model>
    #####################

    ######################
    # <Calculate Metrics>
    ######################
    metrics['cv_mae'] = results_mean["test-MAE-mean"]
    metrics['cv_rmse'] = results_mean["test-RMSE-mean"]
    metrics['cv_mape'] = results_mean["test-MAPE-mean"]
    metrics['cv_r2'] = results_mean["test-R2-mean"]
    
    metrics['test_mae'] = model.evals_result_["validation"]["MAE"][-1]
    metrics['test_rmse'] = model.evals_result_["validation"]["RMSE"][-1]
    metrics['test_r2'] = model.evals_result_["validation"]["R2"][-1]
    metrics['test_mape'] = model.evals_result_["validation"]["MAPE"][-1]
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
    plot_feature_importance(model.get_feature_importance(), feature_names)
    plt.savefig(feat_importance_path, dpi=100, bbox_inches='tight')
    artifacts.append(feat_importance_path)
    plt.close()

    for key in model.evals_result_["learn"].keys():
        metric_plot_path = f"{base_path}/{model_name}{run_name}_learning_curve_{key}.png"
        train_loss = model.evals_result_["learn"][key]
        test_loss = model.evals_result_["validation"][key]
        plt.plot(train_loss)
        plt.plot(test_loss)
        plt.xlabel("Iterations")
        plt.ylabel(f"{key}")
        plt.legend(["Training Loss", "Validation Loss"])
        plt.savefig(metric_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        artifacts.append(metric_plot_path)
    ######################
    # </Evaluation Plots>
    ######################
    return model, metrics, artifacts, output_parameters