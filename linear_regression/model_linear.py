from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score


def create_linear_model(params: dict):
    model = make_pipeline(
        SimpleImputer(strategy="mean"),
        # PolynomialFeatures(degree=params["polynomialfeatures__degree"]),
        LinearRegression(**params),
    )
    return model


def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    r2 = round(r2_score(y_test, y_pred), 3)
    slope = model["linearregression"].coef_
    intercept = model["linearregression"].intercept_

    return r2, slope, intercept
