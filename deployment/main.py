import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import List

import sys

sys.path.append("../scripts/")
import procesing


class Postition(BaseModel):
    """Tip model based on seaborn tip"""

    competition_id: float
    home_club_id: float
    away_club_id: float
    home_club_position: float
    away_club_position: float


app = FastAPI()


@app.on_event("startup")
async def load_model():
    global model
    model = joblib.load("model.joblib")


@app.on_event("startup")
async def load_data():
    global static_table
    static_table = procesing.read_data_from_file("data.json")


@app.get("/static_table")
def get_static_table():
    return static_table


@app.get("/static_table/club_id={club_id}")
def get_team_by_id(club_id: str):
    for item in static_table:
        if item["club_id"] == int(club_id):
            return item
    return {"message": "Team not found"}


@app.post("/predict")
def predict(pos: Postition):
    data = pd.DataFrame([dict(pos)])
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}


# input team name > get table position ant stats, post stats to predict
