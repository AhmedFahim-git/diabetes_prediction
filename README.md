# Diabetes Prediction

Here we are using the [dibetes prediction dataset](https://www.kaggle.com/datasets/joebeachcapital/diabetes-factors). We are taking into account a patient's Glucose, BMI etc. to determine if a patient will live. The prediction variable is Outcome, which is encoded as 1 if "dead" and 0 if "alive"

Here we want to deploy the model to be availble for batch predictions

## Cloud

The entire project is set up using docker compose. So it can be easity deployed. It uses localstack as well for S3 bucket for MLFlow.

No IaC tools were used yet.

## Experiment tracking and model registry

Experiment tracking and model registry are included in the all_logic/model.py file. And we start the MLflow server in the all_logic/script.sh script.

During prediction we use the model in the Producion stage

## Workflow orchestration

We are using Prefect for workflow orchestration. The flow is in the all_logic/model.py file. The deployment is being made and run in the all_logic/script.sh script.

## Model deployment

The model is deployed as a docker contained using the all_logic/deploy.py file. The docker container exposes the port 8000 for the model. We can send a post request to the /predict endpoint to get the model predictions from the model in the Production stage.

## Model monitoring

All the model predictions are stored in a `prediction` table in the Postgres database. That is used monitor model metrics and performance over time. Moreover, in the prediction script predictions/predictions/predict.py, it is set such that if the overall accuracy drops below 80% it will trigger a model retraining by sending a request on the /retrain endpoint.

## Reproducibility

The code is run by using the command

```bash
bash ./run.sh
```

The tests are run by using the command

```bash
bash ./run_tests.sh
```

## Extras

Unit tests and integration tests are in the tests directory.
Black was used as the code formatter.
Pre-commit hooks were used.
