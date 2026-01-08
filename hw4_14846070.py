import numpy as np

def scale_to_range(X: np.ndarray, to_range=(0, 1), byrow=False):
    """
    Min-max normalization

    Parameters
    ----------
    X : np.ndarray
        1D or 2D array
    to_range : tuple, default (0,1)
        Desired range (a, b)
    byrow : bool, default False
        If X is 2D:
        - False: column-wise normalization
        - True : row-wise normalization

    Returns
    -------
    Y : np.ndarray
        Scaled array
    """
    a, b = to_range
    X = np.asarray(X, dtype=float)
    Y = np.zeros_like(X)

    # ---------- 1D ----------
    if X.ndim == 1:
        xmin = X.min()
        xmax = X.max()
        Y = (X - xmin) / (xmax - xmin) * (b - a) + a
        return Y

    # ---------- 2D ----------
    if byrow:
        # row-wise
        for i in range(X.shape[0]):
            row = X[i, :]
            xmin = row.min()
            xmax = row.max()
            Y[i, :] = (row - xmin) / (xmax - xmin) * (b - a) + a
    else:
        # column-wise
        for j in range(X.shape[1]):
            col = X[:, j]
            xmin = col.min()
            xmax = col.max()
            Y[:, j] = (col - xmin) / (xmax - xmin) * (b - a) + a

    return Y
A = np.array([1., 2.5, 6., 4., 5.])
print(scale_to_range(A))
A = np.array([
    [ 1, 12,  3,  7,  8],
    [ 5, 14,  1,  5,  5],
    [ 4, 11,  4,  1,  2],
    [ 3, 13,  2,  3,  5],
    [ 2, 15,  6,  3,  2]
])
print(scale_to_range(A))
A = np.array([
    [1, 2, 3, 4, 5],
    [5, 4, 1, 2, 3],
    [3, 5, 4, 1, 2]
])
print(scale_to_range(A, byrow=True))
