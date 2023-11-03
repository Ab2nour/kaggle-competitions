from datetime import datetime

import pandas as pd
import xgboost
from config import seed
from cv_strategy import create_cv_strategy
from sklearn.model_selection import GridSearchCV, StratifiedKFold

grid_search_folder = "data/results/hyper-parameter-tuning/grid-search"


def gs(
    x_train,
    y_train,
    model,
    model_params: dict,
    cv,
    scoring: str = "roc_auc",
    n_jobs: int = -2,
):
    model_name = model.__class__.__name__

    clf = GridSearchCV(
        estimator=model,
        param_grid=model_params,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        cv=cv,
    )

    clf.fit(x_train, y_train)
    results = pd.DataFrame(clf.cv_results_)

    date = str(datetime.now())
    results.to_csv(f"{grid_search_folder}/{model_name}-{date}.csv", index=False)


def gs_xgboost(x_train, y_train):
    model = xgboost.XGBClassifier(n_jobs=-2, random_state=0)

    model_params = {
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 500, 1000],
        "colsample_bytree": [0.3, 0.7],
    }

    kfold = create_cv_strategy(seed)

    gs(x_train, y_train, model, model_params, kfold)
