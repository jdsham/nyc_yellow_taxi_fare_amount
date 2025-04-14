from sklearn.neural_network import MLPRegressor
import numpy as np
import argparse
import matplotlib.pyplot as plt
from custom_funcs import calc_metrics_sklearn, plot_residuals, plot_true_vs_pred, plot_residual_descriptive_stats, plot_errors_to_features
import optuna


def plot_train_val_loss(training_loss, validation_loss, base_path, artifacts):
    # Plot the data
    plt.close()
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 5

    x_train = list(range(0,len(training_loss)))
    y_train = training_loss

    x_val = list(range(0,len(validation_loss)))
    y_val = validation_loss

    path = f"{base_path}/train_val_loss.png"
    artifacts.append(path)
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Epochs vs Loss")
    plt.ylabel("Training Loss / Validation R2")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.plot(x_train, y_train, label="Training Loss")
    plt.plot(x_val, y_val, label="Validation R2")
    plt.autoscale(axis="x", tight=True)
    plt.legend()
    plt.savefig(path)
    plt.close()
    return artifacts


def objective(trial, args, X_train, y_train):
    hidden_layer_sizes = trial.suggest_int("hidden_layer_sizes", 10,100, step=5)
    alpha = trial.suggest_float("alpha", 0.00001, 0.001, step=0.00001)
    learning_rate_init = trial.suggest_float("learning_rate_init", 0.001, 0.1, step=0.001)
    model_params = {"random_state": args.random_state,
                    "hidden_layer_sizes":hidden_layer_sizes,
                    "alpha": alpha,
                    "learning_rate_init": learning_rate_init,
                    "early_stopping": True,
                    "validation_fraction":0.01,
                    }
    model = MLPRegressor(**model_params)
    model.fit(X_train, y_train)
    results = model.best_validation_score_
    return results


def run_mlp_sklearn(X_train:np.array, y_train:np.array, X_test:np.array, y_test:np.array, args:argparse.ArgumentParser, base_path:str, run_name:str, feature_names:list, metrics:dict, artifacts:list, W_train:np.array=None) -> tuple:
    """Uses MLP from SKLearn to train and evaluate model performance with validation data.
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
    output_parameters = dict()
    ####################
    # <Train the Model>
    ####################
    output_parameters = dict()
    model_name = args.model

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args, X_train, y_train), n_trials=30)
    trial = study.best_trial
    best_params = trial.params
    output_parameters = best_params

    model_params = {"early_stopping": True, "random_state": args.random_state} | best_params
  
    model = MLPRegressor(**model_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #####################
    # </Train the Model>
    #####################

    ######################
    # <Calculate Metrics>
    ######################
    metrics["cv_r2"] =  model.best_validation_score_
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
    # Validation Curves
    artifacts = plot_train_val_loss(model.loss_curve_, model.validation_scores_, base_path, artifacts)
    ###################
    # </Metric Curves>
    ###################

    return model, metrics, artifacts, output_parameters