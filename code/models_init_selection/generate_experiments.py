
# Generate a list of experiments to run by generating arg combinations
# Each experiment will be excuted via bash script

experiments = []
data = "/home/joe/datum/fare_amount_init_clean_6_8_2023.parquet"
experiment_name = "fare_amount_initial_selection"
features = ["trip_distance,trip_duration_min,r,theta"]
run_name = "initial_optimized"
random_state = None
scalers = ["asis"]
models = ["lgbm"]

model_params = '{"num_leaves":163,"num_iterations":450,"min_data_in_leaf":50,"learning_rate":0.068}'


# Train 10 different LGBM models without random state set.
# Then perform CV to see average performance
for i in range(0,10):
    for model in models:
        for feature in features:
            for scaler in scalers:
                code = f"train_model.py --run_name {run_name} --model {model} --data {data} --model_params {model_params} --features {feature} --scaler {scaler} --experiment_name {experiment_name}"
                experiments.append(code)


with open("experiments.txt", "w") as file:
    for item in experiments:
        file.write(item + "\n")