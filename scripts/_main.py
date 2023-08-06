import sys

sys.path.append("./scripts/")
sys.path.append("./linear_regression/")
sys.path.append("./random_forest/")

from transformer import DataTransformer
import pandas as pd
import procesing
import config
import joblib

import model_forest
import model_config_forest
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.linear_model import LinearRegression


def random_forest():
    df = pd.read_csv("./data/games.csv")
    transformer = DataTransformer()
    X = transformer.fit_transform(df)
    df_to_model, df_random = procesing.split_data_random_and_Xy(X)

    X, y = procesing.split_X_y(df_to_model)
    tts = train_test_split(X, y, stratify=y, test_size=0.3)

    X_train, X_test, y_train, y_test = tts

    model = model_forest.create_random_forest(params=model_config_forest.params)

    model.fit(X_train, y_train)
    joblib.dump(model, "model.joblib")

    score = model.score(X_test, y_test)

    feature_importances = model.named_steps[
        "randomforestclassifier"
    ].feature_importances_

    for feature, importance in zip(X_train.columns, feature_importances):
        print(f"{feature}: {importance}")

    return model, score, X, y


if "__main__" == __name__:
    model, score, X, y = random_forest()
    print(X)

    # SGDRegressor(loss='squared_loss', learning_rate='constant', eta0=0.01, random_state=42)
