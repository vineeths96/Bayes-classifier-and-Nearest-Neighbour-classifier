import numpy as np

MAX_LINES = 1000
DIMENSIONS = 20


def load_from_file(X, Y, filename):
    file = open(filename)
    idx = 0
    for line in file:
        line_list = line.rstrip().split(',')
        line_list = list(filter(None, line_list))

        if len(line_list) == 0:
            continue

        for col_ind in range(DIMENSIONS):
            X[idx, col_ind] = float(line_list[col_ind])

        Y[idx] = int(line_list[20])
        idx += 1


def load_data(choice):
    X_train = np.zeros([MAX_LINES, DIMENSIONS])
    X_test = np.zeros([MAX_LINES, DIMENSIONS])
    Y_train = np.zeros(MAX_LINES)
    Y_test = np.zeros(MAX_LINES)

    filename = 'datasets/P2' + choice +'_train_data_20D.txt'
    load_from_file(X_train, Y_train, filename)

    filename = 'datasets/P2' + choice +'_test_data_20D.txt'
    load_from_file(X_test, Y_test, filename)

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    load_data()