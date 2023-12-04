from sklearn.model_selection import KFold


def create_cv_strategy(seed: int, n_splits: int = 5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return kfold
