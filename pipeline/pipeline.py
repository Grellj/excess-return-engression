import numpy as np
# This serves to transform X_past into a two-dimensional representation to be used in engression
def X_unifier (X_past, X_weekday, X_std, X_all):
    # Since we will use these four inputs based on the CGM input builder it can be assumed that they have an equal number of instances
    X_past_flat = X_past.reshape(X_past.shape[0], -1)
    X_engressable = np.concatenate([X_past_flat, X_weekday, X_std, X_all],
    axis=1
    )
    X_engressable = X_engressable.astype(np.float32)
    return X_engressable

# prepares Y to be used in engression by removing the superfluous cgm target dimension
def Y_unifier (Y):
    Y_engressable = np.squeeze(Y, axis = 2)
    Y_engressable= Y_engressable.astype(np.float32)
    return Y_engressable
