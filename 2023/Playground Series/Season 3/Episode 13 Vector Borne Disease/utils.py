import numpy as np


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def map3_score(y_true, y_pred):
    """Renvoie le score MAP@3 des données prédites y_pred par rapport aux vraies données y_true."""
    sorted_prediction_ids = np.argsort(-y_pred, axis=1)
    top_3_prediction_ids = sorted_prediction_ids[:, :3]

    return mapk(y_true.reshape(-1, 1), top_3_prediction_ids, k=3)


def top3_predictions(model, X, label_encoder):
    y_pred = model.predict_proba(X)

    sorted_prediction_ids = np.argsort(-y_pred, axis=1)
    top_3_prediction_ids = sorted_prediction_ids[:, :3]

    original_shape = top_3_prediction_ids.shape
    top_3_predictions_array = label_encoder.inverse_transform(
        top_3_prediction_ids.reshape(-1, 1)
    )
    top_3_predictions_array = top_3_predictions_array.reshape(original_shape)

    return top_3_predictions_array
