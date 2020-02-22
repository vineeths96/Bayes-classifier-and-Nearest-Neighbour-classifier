from problem_4.accuracy import accuracy


def model_test(model, X_test, Y_test):
    Y_pred = model.predict(X_test)

    BOW_accuracy = accuracy(Y_pred, Y_test)

    return BOW_accuracy


if __name__ == "__main__":
    model_test()