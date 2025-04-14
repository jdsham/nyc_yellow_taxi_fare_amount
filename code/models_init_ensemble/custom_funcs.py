import numpy as np
import pandas as pd
from pandas.plotting import table
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Callable
import json
import mlflow
plt.switch_backend('agg')

base_path = "/home/joe/datum/experiments"

####################
#<Input Arg Parsing>
####################
def list_of_strings(arg):
    return arg.split(',')

def parse_input_args(parser:argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds parser arguements to the ArgumentParser object.
    Used for CLI based runs for experiments

    Args:
        parser (argparse.ArgumentParser): the argument parser

    Returns:
        argparse.ArgumentParser: the argument parser with added arguements
    """
    parser.add_argument("--run_name", type=str, default="stock", help="A given run name to help describe the experiment being performed. Default is 'stock'.")
    parser.add_argument("--models_for_ensemble", type=list_of_strings, required=True, help="Specify the models to use for the ensemble. Supported models are lgbm, xgboost, linear_regression, linear_svr, and mlp_sklearn.")
    parser.add_argument("--data", type=str, help="path to input data for train-test split.")
    parser.add_argument("--ensemble_type", type=str, required=True, help="The ensemble type to use. Specify either stacking or voting.")
    parser.add_argument("--weights", type=str, default=None, help="The name of the column containing weights")
    parser.add_argument("--features", type=list_of_strings, default="trip_distance,trip_duration_min", help="Which features should be included for training.")
    parser.add_argument("--scaler", type=str, required=False, default="minmax", help="How to scale numerical features.")
    parser.add_argument("--random_state", type=int, required=False, default=42, help="Specify the random states to use.")
    parser.add_argument("--cv", type=int, required=False, default=5, help="Specify number of CV folds to perform.")    
    parser.add_argument("--experiment_name", type=str, required=False, default=None, help="Name of the data science experiment.")
    parser.add_argument("--registered_model_name", type=str, required=False, default=None, help="Name of the model to register.")
    return parser
#####################
#</Input Arg Parsing>
#####################


####################
# <Feature Scalers>
####################
scaler_dict = {
    "minmax": MinMaxScaler,
    "std": StandardScaler,
    "robust": RobustScaler,
}
#####################
# </Feature Scalers>
#####################


######################
# <Calculate Metrics>
######################
def calc_cv_metrics_sklearn(metrics:dict, cv:dict) -> dict:
    """Computes cross validation metrics from Sklearn's cross_validate function and adds the metric names and values to the metrics dictionary.

    Args:
        metrics (dict): dictionary storying metric names and values
        cv (dict): the cross_validate scores dictionary returned from calling cross_validate

    Returns:
        dict: the updated metrics dictionary containing the computed metrics
    """
    metrics['cv_mae'] = np.mean(cv["test_neg_mean_absolute_error"])*-1
    metrics['cv_rmse'] = np.mean(cv["test_neg_root_mean_squared_error"])*-1
    metrics['cv_mape'] = np.mean(cv["test_neg_mean_absolute_percentage_error"])*-1
    metrics['cv_r2'] = np.mean(cv["test_r2"])
    return metrics

def calc_metrics_sklearn(metrics:dict, y_true:np.array, y_pred:np.array, data_type="test") -> dict:
    """Uses Sklearn metrics to calculate metrics based on the y_true and y_pred values.

    Args:
        metrics (dict): dictionary storying metric names and values
        y_true (np.array): the true values of the target that the model is trying to predict
        y_pred (np.array): the predicted values of the target generated from the model
        data_type (str, optional): specify the data type such as "test" or "train". Defaults to "test".

    Returns:
        dict: the updated metrics dictionary containing the computed metrics
    """
    metrics[f'{data_type}_mae'] = mean_absolute_error(y_true, y_pred)
    metrics[f'{data_type}_rmse'] = root_mean_squared_error(y_true, y_pred)
    metrics[f'{data_type}_mape'] = mean_absolute_percentage_error(y_true, y_pred)
    metrics[f'{data_type}_r2'] = r2_score(y_true, y_pred)
    return metrics
#######################
# </Calculate Metrics>
#######################

#################
# <Metric Plots>
#################
def plot_residuals(y_true:np.array, y_pred:np.array, model_name:str, run_name:str, base_path:str, artifacts:list) -> list:
    """Plots residuals from predictions.
    Produces a scatter plot, a boxen plot, and a box plot of residuals

    Args:
        y_true (np.array): The true values being predicted
        y_pred (np.array): Tge predicted values
        model_name (str): The name of the model 
        run_name (str): The name of the run
        base_path (str): The base path to save the plots
        artifacts (list): A list containing paths of artifacts to be saved in MLFlow

    Returns:
        list: Updated list containing paths of artifacts to be saved in MLFlow. Adds the paths for the plots generated and saved.
    """
    # Errors / Residuals
    # Errors / Residuals - Scatter
    errors_fig_path = f"{base_path}/{model_name}_{run_name}_errors.png"
    errors = [np.abs(y_true[i] - y_pred[i]) for i in range(0, y_true.shape[0])]
    plt.scatter(y_true, errors)
    plt.title(f"True Values vs. Residuals for {model_name} - {run_name}")
    plt.xlabel("True Values")
    plt.ylabel("Residuals")
    plt.savefig(errors_fig_path, dpi=100, bbox_inches='tight')
    artifacts.append(errors_fig_path)
    plt.close()

    # Errors / Residuals - Boxen Plot
    errors_boxen_fig_path = f"{base_path}/{model_name}_{run_name}_errors_boxen.png"
    sns.boxenplot(x=errors)
    plt.title(f"Boxen Plot of Residuals for {model_name} - {run_name}")
    plt.xlabel("Errors")
    plt.savefig(errors_boxen_fig_path, dpi=100)
    artifacts.append(errors_boxen_fig_path)
    plt.close()

    # Errors / Residuals - Boxplot
    errors_box_fig_path = f"{base_path}/{model_name}_{run_name}_errors_box.png"
    sns.boxplot(x=errors)
    plt.title(f"Box Plot of Residuals for {model_name} - {run_name}")
    plt.xlabel("Errors")
    plt.savefig(errors_box_fig_path, dpi=100)
    artifacts.append(errors_box_fig_path)
    plt.close()
    return artifacts


def plot_true_vs_pred(y_true:np.array, y_pred:np.array, model_name:str, run_name:str, base_path:str, artifacts:list) -> list:
    """Plots residuals from predictions.
    Produces a scatter plot, a boxen plot, and a box plot of residuals

    Args:
        y_true (np.array): The true values being predicted
        y_pred (np.array): Tge predicted values
        model_name (str): The name of the model 
        run_name (str): The name of the run
        base_path (str): The base path to save the plots
        artifacts (list): A list containing paths of artifacts to be saved in MLFlow

    Returns:
        list: Updated list containing paths of artifacts to be saved in MLFlow. Adds the paths for the plots generated and saved.
    """
    # Errors / Residuals
    # Errors / Residuals - Scatter
    true_vs_pred_fig_path = f"{base_path}/{model_name}_{run_name}_true_vs_pred.png"
    plt.scatter(y_true, y_pred)
    x_min, x_max = plt.xlim()
    plt.ylim(x_min, x_max)
    plt.title(f"True Values vs. Predicted Values for {model_name} - {run_name}")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.savefig(true_vs_pred_fig_path, dpi=100, bbox_inches='tight')
    artifacts.append(true_vs_pred_fig_path)
    plt.close()
    return artifacts


def plot_residual_descriptive_stats(y_true:np.array, y_pred:np.array, model_name:str, run_name:str, base_path:str, artifacts:list) -> list:
    """Plots residual statistical descriptions and distributions from predictions.
    Produces histogram of residuals and table of descriptive statistics.

    Also does the same for raw errors (difference = y_pred - y_true)

    Args:
        y_true (np.array): The true values being predicted
        y_pred (np.array): Tge predicted values
        model_name (str): The name of the model 
        run_name (str): The name of the run
        base_path (str): The base path to save the plots
        artifacts (list): A list containing paths of artifacts to be saved in MLFlow

    Returns:
        list: Updated list containing paths of artifacts to be saved in MLFlow. Adds the paths for the plots generated and saved.
    """
    # Residuals
    residual_path = f"{base_path}/{model_name}_{run_name}_residual_descriptive_stats.png"    
    residuals = pd.DataFrame({"residuals":np.abs(y_true-y_pred)})
    # Round to 2 decimal places since these values refer to currency
    residuals_stats = residuals.describe().round(2)
    # Plot the results as a table
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    table(ax, residuals_stats, loc='center')
    plt.title(f"Descriptive Statistics of Residuals for {model_name} - {run_name}")
    plt.savefig(residual_path, dpi=100, bbox_inches='tight')
    plt.close()
    artifacts.append(residual_path)

    # Histogram
    residual_path = f"{base_path}/{model_name}_{run_name}_residual_distribution.png"   
    sns.histplot(data=residuals, x="residuals")
    plt.title(f"Distribution of Residuals for {model_name} - {run_name}")
    plt.savefig(residual_path, dpi=100, bbox_inches='tight')
    plt.close()
    artifacts.append(residual_path)

    # Raw Errors
    error_path = f"{base_path}/{model_name}_{run_name}_error_descriptive_stats.png"
    errors = pd.DataFrame({"errors":y_pred - y_true})
    # Round to 2 decimal places since these values refer to currency
    errors_stats = errors.describe().round(2)
    # Plot the errors as a table
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    table(ax, errors_stats, loc='center')
    plt.title(f"Descriptive Statistics of Errors for {model_name} - {run_name}")
    plt.savefig(error_path, dpi=100, bbox_inches='tight')
    plt.close()
    artifacts.append(error_path)

    # Histogram
    error_path = f"{base_path}/{model_name}_{run_name}_error_distribution.png"   
    sns.histplot(data=errors, x="errors")
    plt.title(f"Distribution of Errors for {model_name} - {run_name}")
    plt.savefig(error_path, dpi=100, bbox_inches='tight')
    plt.close()
    artifacts.append(error_path)
    return artifacts


def plot_errors_to_features(X_test:np.array, y_true:np.array, y_pred:np.array, feature_names:list, model_name:str, run_name:str, base_path:str, artifacts:list) -> list:
    """Plots the residuals and raw errors to the features that were used to train and test the model.
    Will plot trip distance vs trip duration and then r vs theta.

    Args:
        X_test (np.array): Values that were used as test data for the model
        y_true (np.array): The true values that the model tried to predict
        y_pred (np.array): The values that the model predicted
        feature_names (list): Names of the features for the input test data
        model_name (str): The name of the model 
        run_name (str): The name of the run
        base_path (str): The base path to save the plots
        artifacts (list): A list containing paths of artifacts to be saved in MLFlow

    Returns:
        list: Updated list containing paths of artifacts to be saved in MLFlow. Adds the paths for the plots generated and saved.
    """
    df = pd.DataFrame(X_test, columns=feature_names)
    residuals = np.abs(y_true - y_pred)
    errors = y_pred - y_true
    df["residuals"] = residuals
    df["errors"] = errors
    bins = list(range(0,25,5))
    bin_labels = bins[1:]
    df["residuals_bin"] = pd.cut(df["residuals"], bins=bins, labels=bin_labels)
    df["errors_bin"] = pd.cut(df["errors"], bins=bins, labels=bin_labels)
    df = df.sort_values(by="errors", ascending=True)

    if "trip_distance" in feature_names and "trip_duration_min" in feature_names:
        df = df.sort_values(by="residuals", ascending=True)
        residual_path = f"{base_path}/{model_name}_{run_name}_residuals_distance_vs_duration.png"    
        sns.scatterplot(data=df, x="trip_distance", y="trip_duration_min", hue="residuals_bin")
        plt.legend(loc='upper right')
        plt.title(f"Residuals for {model_name} - {run_name}")
        plt.xlabel("Trip Distance")
        plt.ylabel("Trip Duration")
        plt.savefig(residual_path, dpi=100, bbox_inches='tight')
        plt.close()
        artifacts.append(residual_path)

        df = df.sort_values(by="errors", ascending=True)
        error_path = f"{base_path}/{model_name}_{run_name}_errors_distance_vs_duration.png"    
        sns.scatterplot(data=df, x="trip_distance", y="trip_duration_min", hue="errors_bin")
        plt.legend(loc='upper right')
        plt.title(f"Errors for {model_name} - {run_name}")
        plt.xlabel("Trip Distance")
        plt.ylabel("Trip Duration")
        plt.savefig(error_path, dpi=100, bbox_inches='tight')
        plt.close()
        artifacts.append(error_path)

    if "r" in feature_names and "theta" in feature_names:
        df = df.sort_values(by="residuals", ascending=True)
        residual_path = f"{base_path}/{model_name}_{run_name}_residuals_r_vs_theta.png"    
        sns.scatterplot(data=df, x="r", y="theta", hue="residuals_bin")
        plt.legend(loc='upper right')
        plt.title(f"Residuals for {model_name} - {run_name}")
        plt.xlabel("R")
        plt.ylabel("Theta")
        plt.savefig(residual_path, dpi=100, bbox_inches='tight')
        plt.close()
        artifacts.append(residual_path)

        df = df.sort_values(by="errors", ascending=True)
        error_path = f"{base_path}/{model_name}_{run_name}_errors_r_vs_theta.png"    
        sns.scatterplot(data=df, x="r", y="theta", hue="errors_bin")
        plt.legend(loc='upper right')
        plt.title(f"Errors for {model_name} - {run_name}")
        plt.xlabel("R")
        plt.ylabel("Theta")
        plt.savefig(error_path, dpi=100, bbox_inches='tight')
        plt.close()
        artifacts.append(error_path)

    return artifacts
##################
# </Metric Plots>
##################



############################
# <Model Logger for MLFlow>
############################
def get_model_logger(model_name:str) -> Callable:
    return mlflow.sklearn.log_model
#############################
# </Model Logger for MLFlow>
#############################