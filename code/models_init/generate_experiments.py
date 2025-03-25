
# Generate a list of experiments to run by generating arg combinations
# Each experiment will be excuted via bash script

experiments = []
train_data = "/home/joe/datum/fare_amount_init_6_8_2023.parquet"
test_data = "/home/joe/datum/fare_amount_init_9_2023.parquet"
experiment_name = "fare_amount_init3"
random_state = 42
remove_outliers = ["--remove_outliers_train --remove_outliers_test", "--remove_outliers_train", ""]
scalers = ["asis"]
models = ["lgbm", "xgboost", "catboost", "rf_sklearn", "rf_lgbm", "dart_lgbm"]
for model in models:
    for outliers in remove_outliers:
        for scaler in scalers:
            code = f"train_model.py --model {model} --train_data {train_data} --test_data {test_data} {outliers} --scaler {scaler} --random_state {random_state} --experiment_name {experiment_name}"
            code = code.replace("  "," ")
            #experiments.append(code)


scalers = ["std", "minmax", "robust"]
#models = ["linear_regression", "lasso", "ridge", "elastic_net"]
models = ["huber", "sgd"]

for model in models:
    for outliers in remove_outliers:
        for scaler in scalers:
            code = f"train_model.py --model {model} --train_data {train_data} --test_data {test_data} {outliers} --scaler {scaler} --random_state {random_state} --experiment_name {experiment_name}"
            code = code.replace("  "," ")
            experiments.append(code)



with open("experiments.txt", "w") as file:
    for item in experiments:
        file.write(item + "\n")