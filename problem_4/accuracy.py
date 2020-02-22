import numpy as np


def accuracy(Y_pred, Y_test):
    accuracy_vector = (Y_pred == Y_test)
    accuracy = np.sum(accuracy_vector)/len(accuracy_vector)

    return accuracy


if __name__ == "__main":
    accuracy()