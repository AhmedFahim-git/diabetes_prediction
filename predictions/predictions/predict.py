from __future__ import annotations

import json
from datetime import datetime
from datetime import timezone
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine

basedir = Path(__file__).resolve().parent.parent.parent


def monitor(sqlalchemy_engine):
    all_preds = pd.read_sql_table("predictions", con=sqlalchemy_engine)
    accuracy = accuracy_score(
        y_true=all_preds["Outcome"], y_pred=all_preds["predictions"]
    )
    if accuracy < 0.8:
        print("Retraining")
        r = requests.get("http://127.0.0.1:8000/retrain")
        print(r.text)
    else:
        print("retraining not needed")


def get_predictions(input_df):
    df_list = input_df.to_dict("records")
    res = requests.post(
        "http://127.0.0.1:8000/predict", json={"dataset": json.dumps(df_list)}
    )
    predictions = pd.DataFrame(res.json())
    return predictions


def get_data():
    print(basedir)
    df = pd.read_csv(basedir / "diabetes-vid.csv")
    df["Outcome"] = (df["Outcome"] == "dead").apply(int)
    return df


def make_predictions(my_df):
    my_df = my_df.reset_index(drop=True)
    input_df = my_df.drop(columns=["Outcome"])
    predictions = get_predictions(input_df=input_df)
    final_df = pd.concat([my_df, predictions], axis=1)
    final_df["prediction_time"] = datetime.now(timezone.utc)
    return final_df


def push_preds_to_db(pred_df, engine):
    pred_df.to_sql(name="predictions", if_exists="append", con=engine, index=False)


if __name__ == "__main__":
    engine = create_engine("postgresql://user:example@localhost:5432/my_db")
    df = get_data()
    preds = make_predictions(df)
    push_preds_to_db(preds, engine=engine)
    monitor(engine)
    print(preds.head())
    print(accuracy_score(y_true=preds["Outcome"], y_pred=preds["predictions"]))

# df = pd.read_csv("diabetes-vid.csv")
# df["Outcome"] = (df["Outcome"] == "dead").apply(int)

# df_list = df.drop(columns=["Outcome"]).to_dict("records")

# res = requests.post(
#     "http://127.0.0.1:8000/predict", json={"dataset": json.dumps(df_list)}
# )
# print(res)
# df = pd.concat([df, pd.DataFrame(res.json())], axis=1)
# df["prediction_time"] = datetime.now(timezone.utc)


# engine = create_engine("postgresql://user:example@localhost:5432/my_db")
# df.to_sql(name="predictions", if_exists="replace", con=engine, index=False)
# print(df.head())
# monitor(engine)
