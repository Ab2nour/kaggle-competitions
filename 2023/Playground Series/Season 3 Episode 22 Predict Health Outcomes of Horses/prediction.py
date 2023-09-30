import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from column_names import target
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
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
        "KNeighborsClassifier5": KNeighborsClassifier(),
        "LinearSVC": LinearSVC(random_state=seed),
        "LogisticRegression": LogisticRegression(random_state=seed),
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
        "RandomForestClassifier": RandomForestClassifier(random_state=seed),
        "XGBClassifier": XGBClassifier(random_state=seed),
        "CatBoostClassifier": CatBoostClassifier(random_state=seed, verbose=False),
        "LGBMClassifier": LGBMClassifier(random_state=seed),
    }


def evaluate_models(
    models: dict[str, BaseEstimator],
    prefix: str,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.DataFrame | np.ndarray,
) -> list[list[str]]:
    """Evalue tous les modèles dans `models` et sauvegarde les résultats avec un préfixe `prefix`
    (utile pour distinguer les différentes stratégies de pré-traitement des données)."""
    results = []

    for model_name, model in models.items():
        name = f"{prefix}/{model_name}"
        print(name)

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=10,
            scoring="f1_micro",
        )

        scores_mean = scores.mean()
        scores_std = scores.std()

        results.append(
            [
                name,
                scores_mean,
                scores_std,
            ]
        )

    print(sorted(results, key=lambda x: x[1], reverse=True))

    return results


def make_prediction(
    model: BaseEstimator,  # todo: maybe type isn't 100% accurate
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.DataFrame | np.ndarray,
    X_kaggle: pd.DataFrame | np.ndarray,
    X_preprocessor: ColumnTransformer,  # todo: maybe type isn't 100% accurate
    y_preprocessor: TransformerMixin,  # todo: maybe type isn't 100% accurate
) -> pd.DataFrame:
    """todo"""
    model.fit(X_train, y_train)

    X_kaggle_processed = pd.DataFrame(
        X_preprocessor.transform(X_kaggle),
        columns=X_preprocessor.get_feature_names_out(),
    )

    y_pred = y_preprocessor.inverse_transform(model.predict(X_kaggle_processed))

    return pd.DataFrame(y_pred, index=X_kaggle.index, columns=[target])
