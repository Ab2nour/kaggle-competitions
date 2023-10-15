import numpy as np
import pandas as pd
from column_names import quali_var_binary, quali_var_for_ohe, quanti_var, target
from config import seed
from cv_strategy import create_cv_strategy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, cross_validate


def evaluate_models(
    models: dict[str, BaseEstimator],
    prefix: str,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.DataFrame | np.ndarray,
    nb_cv: int = 10,
    scoring: str = "roc_auc",
) -> list[list[str]]:
    """Evalue tous les modèles dans `models` et sauvegarde les résultats avec un préfixe `prefix`
    (utile pour distinguer les différentes stratégies de pré-traitement des données)."""
    results = []

    kfold = create_cv_strategy(seed, nb_cv)

    for model_name, model in models.items():
        name = f"{prefix}/{model_name}"
        print(name)

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=kfold,
            scoring=scoring,
            n_jobs=-1,
        )

        scores_mean = scores["test_score"].mean()
        scores_median = np.median(scores["test_score"])
        scores_std = scores["test_score"].std()
        scores_min = scores["test_score"].min()

        scores_time = scores["fit_time"].sum() / nb_cv

        results.append(
            [
                name,
                scores_mean,
                scores_median,
                scores_std,
                scores_min,
                scores_time,
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
        # columns=X_preprocessor.get_feature_names_out(), #fixme: redo this line
    )
    raw_predictions = model.predict(X_kaggle_processed)
    y_pred = y_preprocessor.inverse_transform(raw_predictions)

    return pd.DataFrame(y_pred, index=X_kaggle.index, columns=[target])
