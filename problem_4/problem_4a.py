from problem_4.BOW_load_data import load_data
from problem_4.BOW_model_train import model_train
from problem_4.BOW_model_test import model_test


def problem_4a():
    X_train, Y_train, X_test, Y_test = load_data()
    model = model_train(X_train, Y_train)
    accuracy = model_test(model, X_test, Y_test)

    output_file = open('./results/problem_4a.txt', "w")
    output_file.write("Accuracy: {}".format(accuracy))
    output_file.close()


if __name__ == "__main__":
    problem_4a()