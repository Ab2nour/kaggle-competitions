from sklearn.model_selection import StratifiedKFold


def create_cv_strategy(seed: int, n_splits: int = 5):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return kfold
