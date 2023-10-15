import pandas as pd
import xgboost
from sklearn.model_selection import GridSearchCV, StratifiedKFold

grid_search_folder = "data/results/hyper-parameter-tuning/grid-search"


def gs(x_train, y_train, model, model_params, cv, scoring: str = "roc_auc"):
    model_name = model.__class__.__name__

    clf = GridSearchCV(
        estimator=model,
        param_grid=model_params,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        cv=cv,
    )

    clf.fit(x_train, y_train)

    results = pd.DataFrame(clf.cv_results_)

    results.to_csv(f"{grid_search_folder}/{model_name}.csv", index=False)


def gs_xgboost(x_train, y_train):
    model = xgboost.XGBClassifier(n_jobs=-1, random_state=0)

    model_params = {
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 500, 1000],
        "colsample_bytree": [0.3, 0.7],
    }

    kfold = StratifiedKFold(shuffle=True, random_state=0)

    gs(x_train, y_train, model, model_params, kfold)
