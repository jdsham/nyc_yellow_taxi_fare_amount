
# Generate a list of experiments to run by generating arg combinations
# Each experiment will be excuted via bash script

experiments = []
data = "/home/joe/datum/fare_amount_init_clean_6_8_2023_unique.parquet"
experiment_name = "fare_amount_rework"
features = ["trip_distance,trip_duration_min", "trip_distance,trip_duration_min,r,theta", "r,theta"]
run_name = "unique_data"
random_state = 42
scalers = ["asis"]
models = ["lgbm", "xgboost", "catboost", "rf_sklearn", "rf_lgbm", "dart_lgbm"]
for model in models:
    for feature in features:
        for scaler in scalers:
            code = f"train_model.py --run_name {run_name} --model {model} --data {data} --features {feature} --scaler {scaler} --random_state {random_state} --experiment_name {experiment_name}"
            experiments.append(code)


scalers = ["minmax"]
models = ["linear_regression", "lasso", "ridge", "elastic_net", "huber", "sgd"]

for model in models:
    for feature in features:
        for scaler in scalers:
            code = f"train_model.py --run_name {run_name} --model {model} --data {data} --features {feature} --scaler {scaler} --random_state {random_state} --experiment_name {experiment_name}"
            experiments.append(code)



with open("experiments.txt", "w") as file:
    for item in experiments:
        file.write(item + "\n")