from problem_3.bayes import bayes
from problem_3.EM_gaussian import EM_gaussian
from problem_3.MLE_gaussian import MLE_gaussian
from problem_3.bayes_mixture import bayes_mixture
from problem_3.nearest_neighbour import nearest_neighbour
from problem_3.accuracy import accuracy


def problem_3b(X_train, Y_train, X_test, Y_test):
    output_file = open('./results/problem_3b.txt', "w")

    output_file.write("EM Algorithm\n")
    X_class_0 = X_train[Y_train == -1]
    X_class_1 = X_train[Y_train == 1]
    theta_0_EM, weights_0 = EM_gaussian(X_class_0)
    theta_1_EM, weights_1 = EM_gaussian(X_class_1)

    Y_pred_EM = bayes_mixture(theta_0_EM, theta_1_EM, weights_0, weights_1, X_test)
    Y_pred_EM[Y_pred_EM == 0] = -1
    Y_pred_EM[Y_pred_EM == 1] = 1

    GMM_accuracy = accuracy(Y_pred_EM, Y_test)
    output_file.write("Accuracy: {}\n\n".format(GMM_accuracy))


    output_file.write("MLE Gaussian\n")

    X_class_0 = X_train[Y_train == -1]
    X_class_1 = X_train[Y_train == 1]
    mu_estimate_0, sigma_estimate_0 = MLE_gaussian(X_class_0, len(X_class_0))
    mu_estimate_1, sigma_estimate_1 = MLE_gaussian(X_class_1, len(X_class_1))

    theta_0 = [mu_estimate_0, sigma_estimate_0]
    theta_1 = [mu_estimate_1, sigma_estimate_1]

    Y_pred_MLE = bayes(theta_0, theta_1, X_test)
    Y_pred_MLE[Y_pred_MLE == 0] = -1
    Y_pred_MLE[Y_pred_MLE == 1] = 1

    MLE_accuracy = accuracy(Y_pred_MLE, Y_test)
    output_file.write("Accuracy: {}\n\n".format(MLE_accuracy))


    output_file.write("Nearest Neighbour\n")

    Y_pred_NN = nearest_neighbour(X_class_0, X_class_1, X_test)
    Y_pred_NN[Y_pred_NN == 0] = -1
    Y_pred_NN[Y_pred_NN == 1] = 1

    NN_accuracy = accuracy(Y_pred_NN, Y_test)
    output_file.write("Accuracy: {}".format(NN_accuracy))

    output_file.close()


if __name__ == "__main__":
    problem_3b()