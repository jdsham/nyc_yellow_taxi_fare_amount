import pandas as pd
import argparse
import mlflow
from custom_funcs import get_model_to_run, parse_input_args, scaler_dict, base_path, get_model_logger
from sklearn.model_selection import train_test_split
from mlflow.data.sources import LocalArtifactDatasetSource

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

    model_to_run = get_model_to_run(args)

    model, metrics, artifacts, output_parameters = model_to_run(X_train, y_train, X_test, y_test, args, base_path, run_name, feature_names, metrics, artifacts, W_train)
    ###################
    # <Log Parameters>
    ###################
    data_prep_params = {
        "model" : args.model,
        "scaler": args.scaler,
        "random_state": args.random_state,
        "cv": args.cv,
        "n_training_examples": f"{n_train_examples:,}",
        "n_testing_examples": f"{n_test_examples:,}",
        "features" : features,
        "run_name": run_name,
        "weights": args.weights
        }
    params = data_prep_params | output_parameters
    params = params | args.model_params
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
    model_logger = get_model_logger(args.model)
    artifact_path = "model"
    model_logger(model, artifact_path, signature=signature)
    ###############
    # </Log Model>
    ###############

    mlflow.end_run()


if __name__ == "__main__":
    # run main function
    main()