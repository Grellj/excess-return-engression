import numpy as np

# determines which forecast is applied
horizon = 1
# This serves to transform X_past into a two-dimensional representation to be used in engression
def X_unifier (X_past, X_weekday, X_std, X_all):
    # Since we will use these four inputs based on the CGM input builder it can be assumed that they have an equal number of instances
    X_past_flat = X_past.reshape(X_past.shape[0], -1)
    X_engressable = np.concatenate([X_past_flat, X_weekday, X_std, X_all],
    axis=1
    )
    if horizon > 1:
        X_list = X_engressable.tolist()
        del X_list[len(X_list)-(horizon-1):len(X_list)]
        X_engressable = np.array(X_list)
    X_engressable = X_engressable.astype(np.float32)
    return X_engressable

# Prepares X_ablation to be used in Engression
def X_abl_unifier (X_ablation):
    X_abl_engressable = X_ablation.reshape(X_ablation.shape[0], -1)
    if horizon > 1:
        X_list = X_abl_engressable.tolist()
        del X_list[len(X_list)-(horizon-1):len(X_list)]
        X_abl_engressable = np.array(X_list)
    X_abl_engressable = X_abl_engressable.astype(np.float32)
    return X_abl_engressable

# prepares Y to be used in engression by removing the superfluous cgm target dimension
def Y_unifier (Y):
    Y_engressable = np.squeeze(Y, axis = 2)
    if horizon > 1:
        Y_list = Y_engressable.tolist()
        for i in range(0, len(Y_list)-(horizon-1)):
            for j in range (1, horizon):
                Y_list[i].extend(Y_list[i+j])
        del Y_list[len(Y_list)-(horizon-1):len(Y_list)]
        Y_engressable = np.array(Y_list)
    Y_engressable= Y_engressable.astype(np.float32)
    return Y_engressable
