import json
import mlflow
import pandas as pd

from typing import Annotated, Any, Dict, AnyStr, List, Union

from fastapi import Body, FastAPI, Request
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
loaded_model = mlflow.sklearn.load_model(model_uri)


class MyData(BaseModel):
    dataset: str


# @app.post("/")
# def read_root(dataset: Annotated[str, Body()]):
#     print(dataset)
#     my_data = pd.DataFrame(json.loads(dataset))
#     result = loaded_model.predict(my_data)
#     return {"results": result.tolist()}


# @app.post("/")
# async def read_root(request: Request):
#     await dataset = request.body()
#     print(dataset)
#     my_data = pd.DataFrame(json.loads(dataset))
#     result = loaded_model.predict(my_data)
#     return {"results": result.tolist()}

# @app.post("/")
# async def read_root(rarbitrary_json: JSONStructure = None):
#     dataset = rarbitrary_json
#     print(dataset)
#     my_data = pd.DataFrame(json.loads(dataset))
#     result = loaded_model.predict(my_data)
#     return {"results": result.tolist()}


@app.post("/")
async def read_root(my_data: MyData):
    dataset = my_data.dataset
    print(dataset)
    my_data = pd.DataFrame(json.loads(dataset))
    result = loaded_model.predict(my_data)
    return {"results": result.tolist()}
