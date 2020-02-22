import numpy as np


def nearest_class(X_0, X_1, X):
    min_distance = np.inf
    min_class = 0

    for idx in range(len(X_0)):
        dist = np.linalg.norm(X - X_0[idx])
        if dist < min_distance:
            min_distance = dist
            min_class = -1

    for idx in range(len(X_1)):
        dist = np.linalg.norm(X - X_1[idx])
        if dist < min_distance:
            min_distance = dist
            min_class = 1

    return min_class


def nearest_neighbour(X_0, X_1, X_test):
    Y_pred = np.zeros(len(X_test))

    for idx in range(len(X_test)):
        X = X_test[idx]
        Y_pred[idx] = nearest_class(X_0, X_1, X)

    return Y_pred


if __name__ == "__main__":
    nearest_neighbour()
