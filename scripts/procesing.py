import pandas as pd
import config
import matplotlib.pyplot as plt
import json


def split_data_random_and_Xy(df: pd.DataFrame) -> pd.DataFrame:
    random_df = df.sample(n=400)
    df = df.drop(random_df.index)

    return df, random_df


def split_X_y(df: pd.DataFrame):
    X = df[config.COL_TO_SELECT_X]

    y = df[config.COL_TO_SELECT_y]

    return X, y


def draw_regression(X, y, slope, intercept):
    plt.scatter(X, y, label="data")

    plt.plot(X, slope * X + intercept, color="red", label="Linear reg.")

    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def create_competition_table():
    import pandas as pd

    COL_TO_SEL = [
        "competition_id",
        "home_club_name",
        "season",
        "home_club_id",
        "home_club_position",
    ]

    df = pd.read_csv(".\\data\\games.csv")
    df = df.loc[df["season"] == 2022]
    df = df.loc[df["competition_type"] == "domestic_league"]
    df["date"] = pd.to_datetime(df["date"])

    result_df = df.loc[df.groupby(["competition_id", "home_club_id"])["date"].idxmax()]

    final_df = result_df[COL_TO_SEL].sort_values("home_club_position")
    final_df = final_df.rename(
        columns={
            "home_club_name": "club_name",
            "home_club_id": "club_id",
            "home_club_position": "club_position",
        }
    )

    json_data = final_df.to_json(orient="records")

    with open("data.json", "w") as file:
        file.write(json_data)


def read_data_from_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


if "__main__" == __name__:
    create_competition_table()
