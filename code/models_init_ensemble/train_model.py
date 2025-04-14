
# Stacking Ensemble
# Ensemble
    # Light GBM blender


# Voting Ensemble


# Input parameters:
#   Data Prep
#   Which models to use in ensemble
#   Model parameters
#   Ensemble to use
#   Ensemble parameters
#   





import pandas as pd
import argparse
import mlflow
from custom_funcs import parse_input_args, scaler_dict, base_path
from sklearn.model_selection import train_test_split
from mlflow.data.sources import LocalArtifactDatasetSource
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import VotingRegressor, StackingRegressor
from ensemble_base import run_ensemble

def main() -> None:
    """Runs the main experiment
    """
    # Things that must be defined
    metrics = {}
    artifacts = []


    # Input Arguments
    parser = argparse.ArgumentParser()
    parser = parse_input_args(parser)
    args = parser.parse_args()

    # Define consants
    target = "fare_amount"
    features =  args.features
    run_name = args.run_name
    
    #####################
    # <Prepare the Data>
    #####################
    # Load the data
    df = pd.read_parquet(args.data)

    source = LocalArtifactDatasetSource(args.data)
    dataset = mlflow.data.from_pandas(
    df[features + [target]], source=source, name=args.data.split('/')[-1].split('.parquet')[0], targets=target)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=args.random_state)
    
    # Cleanup for memory
    del df
    
    n_train_examples = train_df.shape[0]
    n_test_examples = test_df.shape[0]

    # Handle weights
    if args.weights is not None:
        weight_col = args.weights
        W_train = train_df[weight_col].to_frame()
        weight_scaler = scaler_dict["minmax"]()
        W_train = weight_scaler.fit_transform(W_train).reshape(-1)
        train_df = train_df.drop(columns=weight_col)
        test_df = test_df.drop(columns=weight_col)
    else:
        W_train = None

    # Scale the data
    if args.scaler != "asis":
        scaler = scaler_dict[args.scaler]()
        X_train = scaler.fit_transform(train_df[features])    
        X_test = scaler.transform(test_df[features])
        feature_names = list(scaler.feature_names_in_)
    else:
        X_train = train_df[features].to_numpy()
        X_test = test_df[features].to_numpy()
        feature_names = features

    y_train = train_df[target].to_numpy()
    y_test = test_df[target].to_numpy()
    ######################
    # </Prepare the Data>
    ######################

    ####################
    # <Train the Model>
    ####################
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name=args.experiment_name)
    mlflow.start_run()

    # Create list of models with their respective parameters
    models_for_ensemble = list()
    for model_name in args.models_for_ensemble:
        if model_name == 'lgbm':
            model_to_add = LGBMRegressor(num_leaves=163, min_data_in_leaf=50, num_iterations=450, learning_rate=0.068, random_state=42)
        elif model_name == 'mlp_sklearn':
            model_to_add = MLPRegressor(hidden_layer_sizes=80, alpha=0.00073, learning_rate_init=0.016, random_state=42)
        elif model_name == 'linear_svr':
            model_to_add = LinearSVR(random_state=42)
        elif model_name == "linear_regression":
            model_to_add = LinearRegression(fit_intercept=True, n_jobs=-1)
        else:
            raise RuntimeError(f"Error: specified model was not recognized. Received {model_name}")
        models_for_ensemble.append((model_name, model_to_add))
    

    if args.ensemble_type == "stacking":
        ensemble_model = StackingRegressor(
                estimators=models_for_ensemble,
                final_estimator=LGBMRegressor(random_state=42)
        )
    else:
        ensemble_model = VotingRegressor(models_for_ensemble)

    model, metrics, artifacts = run_ensemble(ensemble_model, X_train, y_train, X_test, y_test, args, base_path, run_name, feature_names, metrics, artifacts, W_train)
    ###################
    # <Log Parameters>
    ###################
    data_prep_params = {
        "scaler": args.scaler,
        "random_state": args.random_state,
        "cv": args.cv,
        "n_training_examples": f"{n_train_examples:,}",
        "n_testing_examples": f"{n_test_examples:,}",
        "features" : features,
        "run_name": run_name,
        "weights": args.weights
        }
    params = data_prep_params | {"models_for_ensemble": args.models_for_ensemble, "ensemble_model": args.ensemble_type}
    mlflow.log_params(params)
    ####################
    # </Log Parameters>
    ####################

    ################
    # <Log Metrics>
    ################
    for key, val in metrics.items():
        mlflow.log_metric(key, val)
    #################
    # </Log Metrics>
    #################

    ##################
    # <Log Artifacts>
    ##################
    for artifact in artifacts:
      mlflow.log_artifact(artifact)
    ###################
    # </Log Artifacts>
    ###################

    ################
    # <Log Dataset>
    ################
    mlflow.log_input(dataset, context="train/test")
    #################
    # </Log Dataset>
    #################

    ##############
    # <Log Model>
    ##############
    signature = mlflow.models.infer_signature(X_train, y_train)
    artifact_path = "model"
    mlflow.sklearn.log_model(model, artifact_path, signature=signature)
    ###############
    # </Log Model>
    ###############

    mlflow.end_run()


if __name__ == "__main__":
    # run main function
    main()