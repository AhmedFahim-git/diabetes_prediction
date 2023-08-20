from __future__ import annotations

import json
import subprocess
from typing import Any
from typing import AnyStr
from typing import Dict
from typing import List
from typing import Union

import mlflow
import pandas as pd
from fastapi import FastAPI
from model import create_engine
from model import get_test_data
from model import get_train_data
from pydantic import BaseModel

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]
app = FastAPI()

df = pd.read_csv("diabetes-vid.csv")
df["Outcome"] = (df["Outcome"] == "dead").apply(int)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
reg_model_name = "diabetes_model"
model_uri = f"models:/{reg_model_name}/Production"


class MyData(BaseModel):
    dataset: str


class MyConnection(BaseModel):
    connect_str: str


@app.post("/predict")
async def read_root(my_data: MyData):
    loaded_model = mlflow.sklearn.load_model(model_uri)
    dataset = my_data.dataset
    my_data = pd.DataFrame(json.loads(dataset))
    result = loaded_model.predict(my_data)
    return {"predictions": result.tolist()}


@app.get("/retrain")
async def retrain():
    subprocess.run(["prefect", "deployment", "run", "main-flow/my_deployment"])
    return {"message": "retrain completed"}


@app.post("/test/get_train")
async def get_train(connection: MyConnection):
    engine = create_engine(connection.connect_str)
    train_df = get_train_data.fn(engine)
    return {"dataset": json.dumps(train_df.to_dict("records"))}


@app.post("/test/get_test")
async def get_test(connection: MyConnection):
    engine = create_engine(connection.connect_str)
    test_df = get_test_data.fn(engine)
    return {"dataset": json.dumps(test_df.to_dict("records"))}
