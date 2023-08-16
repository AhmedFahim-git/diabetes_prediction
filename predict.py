import pandas as pd
import requests
import json

df = pd.read_csv("diabetes-vid.csv")
df["Outcome"] = (df["Outcome"] == "dead").apply(int)

df_list = df.iloc[:3].drop(columns=["Outcome"]).to_dict("records")

res = requests.post("http://127.0.0.1:8000/", json={"dataset": json.dumps(df_list)})
