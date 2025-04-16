# Model Deployment
Now that a model is selected, it is time to deploy it using Docker and FastAPI.

## How to Deploy
The `docker-compose.yml` file can be used locally or where applicable to easily setup and run the model serving container.
Specifically, the contents inside the `/src` directory can be directly uploaded to Huggingface's Spaces to launch a containerized app the serves the model.
The dockerfile is already configured to run with Huggingface's Spaces.

## How to Use
To test that the API works, one can use GET to `/`.
Example: GET to `localhost:7860/`

To use the service, simply submit a POST request to `/predict` with the body `{"trip_distance": (put trip miles here), "trip_duration": (put trip duration in minutes, including decimals)}` and the API will respond with `{"result": (predicted fare amount)}`
Example: POST to `localhost:7860/predict` with body `{"trip_distance": 2.1, "trip_duration": 7.7}` returns `{"result": 11.23}`