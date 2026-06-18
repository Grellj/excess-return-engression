import numpy as np

# Continuous Ranked Probability Score (CRPS)
def crps_sample(y_true, y_pred):
    """
    Compute the Continuous Ranked Probability Score (CRPS).

    Parameters:
        y_true : array, shape(n_examples,)
            True values.
        y_pred : array, shape (n_examples, n_samples)
            Predictive scenarios.

    Returns:
        float: Continuous Ranked Probability Score.
    """
    n_ens = y_pred.shape[1]
    c = 1 / n_ens
    x = np.sort(y_pred, axis=1)
    y = np.expand_dims(y_true, 1)
    a = np.linspace(0.5 * c, 1 - 0.5 * c, num=n_ens)
    a = np.expand_dims(a, 0)
    score = 2 * c * np.sum(((y < x) - a) * (x - y), axis=1)
    
    return score

# Continuous Ranked Probability Score (CRPS) - another version
def crps_sample_old(y_true, y_pred):
    """
    Parameters
    ----------
    y_true : array, shape(n_examples,)
        True values.
    y_pred : array, shape (n_examples, n_samples)
        Predictive scenarios.

    Returns: CRPS
    """
    y_true = np.expand_dims(y_true, 1)

    crps_1 = np.mean(np.abs(y_true - y_pred), axis=1)
    crps_2 = np.zeros(y_true.shape[0])
    
    for i in range(y_pred.shape[0]):
        crps_2[i] = np.mean(np.abs(y_pred[[i],:].T - np.repeat(y_pred[[i],:], y_pred.shape[1], axis=0)),)

    scores = crps_1 - 0.5*crps_2
    print(scores)
    
    return scores
    