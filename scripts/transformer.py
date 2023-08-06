import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer


def add_result(df: pd.DataFrame):
    df = df.assign(
        result=lambda x: np.where(
            x["home_club_goals"] > x["away_club_goals"],
            "home_win",
            np.where(x["home_club_goals"] < x["away_club_goals"], "home_loss", "draw"),
        )
    )

    return df


class DataTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        df_copy = X.copy()
        df_copy = add_result(df_copy)

        cat_transformer = make_pipeline(OrdinalEncoder())

        preprocessor = ColumnTransformer(
            transformers=[
                ("result", cat_transformer, ["result"]),
                ("competition", cat_transformer, ["competition_id"]),
            ]
        )
        transformed_values = preprocessor.fit_transform(df_copy)
        transformed_df = pd.DataFrame(
            transformed_values, columns=["result", "competition_id"]
        )

        df_copy["result"] = transformed_df["result"]
        df_copy["competition_id"] = transformed_df["competition_id"]

        return df_copy
