from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import requests
from sqlalchemy import create_engine

from predictions import predict

base_path = Path("__file__").resolve().parent


@pytest.fixture
def db_init():
    df = pd.read_csv("diabetes-vid.csv")
    df["Outcome"] = (df["Outcome"] == "dead").apply(int)
    engine = create_engine("postgresql://user:example@localhost:5433/my_db")
    train = df.iloc[:5].reset_index(drop=True)
    test = df.iloc[-5:].reset_index(drop=True)
    train.to_sql(name="train", if_exists="replace", con=engine, index=False)
    test.to_sql(name="test", if_exists="replace", con=engine, index=False)
    return {"train_data": train, "test_data": test}


@pytest.fixture
def data_train(db_init):
    return db_init["train_data"]


@pytest.fixture
def data_test(db_init):
    return db_init["test_data"]


def test_get_train_data(data_train):
    res = requests.post(
        "http://127.0.0.1:8000/test/get_train",
        json={"connect_str": "postgresql://user:example@testdb:5432/my_db"},
    )
    df_list = json.loads(res.json()["dataset"])
    actual = pd.DataFrame(df_list)
    pd.testing.assert_frame_equal(data_train, actual)


def test_get_test_data(data_test):
    res = requests.post(
        "http://127.0.0.1:8000/test/get_test",
        json={"connect_str": "postgresql://user:example@testdb:5432/my_db"},
    )
    df_list = json.loads(res.json()["dataset"])
    actual = pd.DataFrame(df_list)
    pd.testing.assert_frame_equal(data_test, actual)


def test_get_data():
    df = pd.read_csv(base_path / "diabetes-vid.csv")
    df["Outcome"] = (df["Outcome"] == "dead").apply(int)
    actual_df = predict.get_data()
    pd.testing.assert_frame_equal(df, actual_df)
