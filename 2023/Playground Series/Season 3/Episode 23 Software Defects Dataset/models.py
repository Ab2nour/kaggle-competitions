from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def create_models(seed: int) -> dict[str, BaseEstimator]:
    """todo"""
    return {
        "DummyClassifier_Uniform": DummyClassifier(
            strategy="uniform", random_state=seed
        ),
        "DummyClassifier_MostFrequent": DummyClassifier(
            strategy="most_frequent", random_state=seed
        ),
        # "KNeighborsClassifier5": KNeighborsClassifier(),
        # "LinearSVC": LinearSVC(random_state=seed),
        "LogisticRegression": LogisticRegression(random_state=seed),
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
        "RandomForestClassifier": RandomForestClassifier(random_state=seed),
        "ExtraTreesClassifier": ExtraTreesClassifier(random_state=seed),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            random_state=seed
        ),
        "XGBClassifier": XGBClassifier(random_state=seed),
        "CatBoostClassifier": CatBoostClassifier(random_state=seed, verbose=False),
        "LGBMClassifier": LGBMClassifier(random_state=seed),
    }
