import requests
import pandas as pd
import json

import sys

sys.path.append("./scripts/")

from transformer import DataTransformer


COL_TO_MODEL = [
    "competition_id",
    "club_id_x",
    "club_id_y",
    "club_position_x",
    "club_position_y",
]


def get_teams_data(home_club_id: int, away_club_id: int):
    # 131
    # 714
    team_home = requests.get(
        f"http://127.0.0.1:8000/static_table/club_id={home_club_id}"
    )
    team_away = requests.get(
        f"http://127.0.0.1:8000/static_table/club_id={away_club_id}"
    )
    json_data_home = team_home.json()
    json_data_away = team_away.json()

    return json_data_home, json_data_away


def data_to_predict(home_club_id: int, away_club_id: int):
    home, away = get_teams_data(home_club_id=home_club_id, away_club_id=away_club_id)

    df_home = pd.DataFrame([home])
    df_away = pd.DataFrame([away])

    full_df = pd.merge(df_home, df_away, how="left", on="competition_id")
    full_df["competition_id"] = 1  # Čia chaltura, nes neapgalvojau CustomTransformerio
    return full_df


def data_to_post(home_club_id: int, away_club_id: int):
    df = data_to_predict(home_club_id=home_club_id, away_club_id=away_club_id)
    df = df[COL_TO_MODEL]
    df = df.rename(
        columns={
            "club_id_x": "home_club_id",
            "club_position_x": "home_club_position",
            "club_id_y": "away_club_id",
            "club_position_y": "away_club_position",
        }
    )

    json_data = df.to_json(orient="records")

    return json.loads(json_data)[0]


def get_prediction(json_data):
    response = requests.post("http://127.0.0.1:8000/predict", json=json_data)
    print(response.json())

    return response.json()


if "__main__" == __name__:
    df = data_to_post(home_club_id=131, away_club_id=714)
    get_prediction(df)
    # print(df)
