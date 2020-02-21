import numpy as np

MAX_LINES = 200
DIMENSIONS = 2

def load_from_file(X, Y, filename):
    file = open(filename)
    idx = 0
    for line in file:
        line_list = line.rstrip().split(',')
        X[idx] = float(line_list[0]), float(line_list[1])
        Y[idx] = int(line_list[2])
        idx += 1


def load_data(choice):
    X_train = np.zeros([MAX_LINES, DIMENSIONS])
    X_test = np.zeros([MAX_LINES, DIMENSIONS])
    Y_train = np.zeros(MAX_LINES)
    Y_test = np.zeros(MAX_LINES)

    filename = 'datasets/P1' + choice +'_train_data_2D.txt'
    load_from_file(X_train, Y_train, filename)

    filename = 'datasets/P1' + choice +'_test_data_2D.txt'
    load_from_file(X_test, Y_test, filename)

    return X_train, Y_train, X_test, Y_test
