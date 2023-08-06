from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


def create_random_forest(params: dict):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        # PolynomialFeatures(degree=params["polynomialfeatures__degree"]),
        RandomForestClassifier(**params),
    )
    return model
