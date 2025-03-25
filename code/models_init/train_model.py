import pandas as pd
import argparse
import mlflow
from custom_funcs import get_model_to_run, RemoveOutliersDistance, RemoveOutliersDuration, parse_input_args, scaler_dict, base_path
from sklearn.pipeline import Pipeline

def main() -> None:
    """Runs the main experiment
    """
    # Things that must be defined
    metrics = {}
    artifacts = []
    run_name = "stock" 

    # Input Arguments
    parser = argparse.ArgumentParser()
    parser = parse_input_args(parser)
    args = parser.parse_args()

    # Define consants
    target = "fare_amount"
    features =  ["trip_distance", "trip_duration_min"]

    #####################
    # <Prepare the Data>
    #####################
    # Load the data
    train_df = pd.read_parquet(args.train_data)
    test_df = pd.read_parquet(args.test_data)

    # Pipeline to remove outliers
    outliers_pipeline = Pipeline([
            ("outliers_distance", RemoveOutliersDistance())
            ,("outliers_duration", RemoveOutliersDuration())
            ])
    
    # If no transformations specified, then just use df as-is
    if args.remove_outliers_train:
        outliers_pipeline.fit(train_df)
        train_df = outliers_pipeline.fit_transform(train_df)
        if args.remove_outliers_test:
            test_df = outliers_pipeline.transform(test_df)
    
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

    metrics, artifacts, output_parameters = model_to_run(X_train, y_train, X_test, y_test, args, base_path, run_name, feature_names, metrics, artifacts)
    ###################
    # <Log Parameters>
    ###################
    data_prep_params = {
        "model" : args.model,
        "remove_outliers_train" : args.remove_outliers_train,
        "remove_outliers_test": args.remove_outliers_test,
        "scaler": args.scaler,
        "random_state": args.random_state,
        "cv": args.cv,
        "train_data": args.train_data.split('/')[-1].split('.')[0],
        "test_data": args.test_data.split('/')[-1].split('.')[0]
        }
    params = data_prep_params | output_parameters
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

    mlflow.end_run()


if __name__ == "__main__":
    # run main function
    main()