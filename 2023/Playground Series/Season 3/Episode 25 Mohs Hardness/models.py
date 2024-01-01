from catboost import CatBoostRegressor
from config import n_jobs
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


def create_models(seed: int) -> dict[str, BaseEstimator]:
    """todo"""
    return {
        # "KNeighborsRegressor5": KNeighborsRegressor(),
        # "LinearSVC": LinearSVC(random_state=seed),
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_jobs=n_jobs, random_state=seed
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(n_jobs=n_jobs, random_state=seed),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            random_state=seed
        ),
        "XGBRegressor": XGBRegressor(n_jobs=n_jobs, random_state=seed),
        # "CatBoostRegressor": CatBoostRegressor(random_state=seed, verbose=False),
        "LGBMRegressor": LGBMRegressor(n_jobs=n_jobs, random_state=seed, verbose=0),
    }
