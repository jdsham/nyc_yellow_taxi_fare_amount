
# Generate a list of experiments to run by generating arg combinations
# Each experiment will be excuted via bash script

experiments = []
data = "/home/joe/datum/fare_amount_init_clean_6_8_2023.parquet"
experiment_name = "fare_amount_initial_ensemble"
feature = "trip_distance,trip_duration_min,r,theta"
run_name = "initial_ensemble"
random_state = 42
scaler = "minmax"
models = "linear_svr,mlp_sklearn,linear_regression,lgbm"
ensemble_types = ["stacking", "voting"]

for ensemble_type in ensemble_types:
    code = f"train_model.py --run_name {run_name} --models_for_ensemble {models} --data {data} --ensemble_type {ensemble_type} --features {feature} --scaler {scaler} --random_state {random_state} --experiment_name {experiment_name}"
    experiments.append(code)



with open("experiments.txt", "w") as file:
    for item in experiments:
        file.write(item + "\n")