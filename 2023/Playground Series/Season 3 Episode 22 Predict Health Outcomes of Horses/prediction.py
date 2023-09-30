import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from column_names import quali_var_binary, quali_var_for_ohe, quanti_var, target
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
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
    raw_predictions = model.predict(X_kaggle_processed)
    y_pred = y_preprocessor.inverse_transform(raw_predictions)

    return pd.DataFrame(y_pred, index=X_kaggle.index, columns=[target])


def create_x_pipeline():
    """todo"""
    quanti_processor = Pipeline(
        steps=[
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
        ]
    )

    quali_ohe_processor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe_encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", sparse_output=False
                ),
            ),
        ]
    )

    quali_binary_processor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=np.nan
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        remainder="passthrough",
        transformers=[
            (
                "quali_ohe",
                quali_ohe_processor,
                quali_var_for_ohe,
            ),
            ("quali_non_ohe", quali_binary_processor, quali_var_binary),
            ("quanti_processor", quanti_processor, quanti_var),
        ],
    )

    return preprocessor
