import numpy as np
import pandas as pd
from column_names import quali_var_binary, quali_var_for_ohe, quanti_var, target
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score


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
            scoring="roc_auc",
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
        # columns=X_preprocessor.get_feature_names_out(), #fixme: redo this line
    )
    raw_predictions = model.predict(X_kaggle_processed)
    y_pred = y_preprocessor.inverse_transform(raw_predictions)

    return pd.DataFrame(y_pred, index=X_kaggle.index, columns=[target])
