import numpy as np
import pandas as pd
from column_names import quali_var_binary, quali_var_for_ohe, quanti_var, target
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


def add_original_data(
        X: np.ndarray, y: np.ndarray, X_original: np.ndarray, y_original: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    X_with_original = np.vstack((X, X_original))
    y_with_original = np.hstack((y, y_original))

    return X_with_original, y_with_original


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    return df


def create_x_pipeline():
    """todo"""
    add_features_transformer = FunctionTransformer(add_features)

    quanti_processor = Pipeline(
        steps=[
            ("imputer", SimpleImputer()),
            # ("scaler", StandardScaler()),
            # ("log_transform", FunctionTransformer(np.log1p)),
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

    column_transformer = ColumnTransformer(
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

    preprocessor = Pipeline(
        steps=[
            ("add_features", add_features_transformer),
            ("column_transformer", column_transformer),
        ]
    )

    return preprocessor


def create_y_pipeline():
    """todo"""
    preprocessor = Pipeline(steps=[
        
            ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    return preprocessor
