import mlflow

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

from mlflow import MlflowClient
from sqlalchemy import create_engine
from prefect import task, flow
import matplotlib

matplotlib.use("Agg")


@task(name="Create train and test tables", retries=3, retry_delay_seconds=10)
def init_db(sqlalchemy_engine):
    df = pd.read_csv("diabetes-vid.csv")
    df["Outcome"] = (df["Outcome"] == "dead").apply(int)
    train, test = train_test_split(
        df, test_size=0.25, random_state=1234, stratify=df["Outcome"]
    )
    train.to_sql(name="train", if_exists="replace", con=sqlalchemy_engine, index=False)
    test.to_sql(name="test", if_exists="replace", con=sqlalchemy_engine, index=False)


@task(name="Fetch train data", retries=3, retry_delay_seconds=10)
def get_train_data(sqlalchemy_engine):
    return pd.read_sql_table("train", con=sqlalchemy_engine)


@task(name="Fetch test data", retries=3, retry_delay_seconds=10)
def get_test_data(sqlalchemy_engine):
    return pd.read_sql_table("test", con=sqlalchemy_engine)


@task(name="Set environment variables")
def set_env_vars():
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:4566"


@task(name="Train and test model", retries=3, retry_delay_seconds=10)
def train_and_test(train_df: pd.DataFrame, test_df=pd.DataFrame):
    X_train = train_df[[col for col in train_df.columns if col != "Outcome"]]
    X_test = test_df[[col for col in test_df.columns if col != "Outcome"]]

    y_train = train_df["Outcome"]
    y_test = test_df["Outcome"]

    client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("diabetes")
    mlflow.sklearn.autolog(
        log_input_examples=True,
        registered_model_name="diabetes_model",
        max_tuning_runs=None,
    )
    clf = RandomForestClassifier()
    parameters = {
        "criterion": ("gini", "log_loss", "entropy"),
        "min_samples_leaf": (1, 5, 10),
    }
    grid_search = GridSearchCV(
        clf,
        parameters,
        scoring=[
            "f1",
            "accuracy",
            "balanced_accuracy",
            "recall",
            "precision",
            "roc_auc",
        ],
        refit="f1",
    )
    grid_search.fit(X_train, y_train)

    grid_search.score(X_test, y_test)

    model_version = client.search_model_versions(
        "name = 'diabetes_model'", order_by=["last_updated_timestamp DESC"]
    )[0].version

    client.transition_model_version_stage(
        name="diabetes_model",
        version=model_version,
        stage="Production",
        archive_existing_versions=True,
    )


@flow
def main_flow():
    engine = create_engine("postgresql://user:example@localhost:5432/my_db")
    init_db(engine)
    train_df = get_train_data(engine)
    test_df = get_test_data(engine)

    set_env_vars()

    train_and_test(train_df=train_df, test_df=test_df)


if __name__ == "__main__":
    main_flow()

# df = pd.read_csv("diabetes-vid.csv")
# df["Outcome"] = (df["Outcome"] == "dead").apply(int)

# X_train, X_test, y_train, y_test = train_test_split(
#     df[[col for col in df.columns if col != "Outcome"]],
#     df["Outcome"],
#     test_size=0.20,
#     random_state=1234,
#     stratify=df["Outcome"],
# )
