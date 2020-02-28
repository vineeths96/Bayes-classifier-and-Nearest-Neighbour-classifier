import random
from problem_2.MLE_gaussian import MLE_gaussian
from problem_2.nearest_neighbour import nearest_neighbour
from problem_2.bayes import bayes
from problem_2.accuracy import accuracy


def problem_2a(X_train, Y_train, X_test, Y_test):
    N_list = [10, 20, 50, 200, 300, 500]
    output_file = open('./results/problem_2a.txt', "w")
    output_file.write("N, bayes, near_neigh\n")

    for N in N_list:
        rand_list = random.sample(range(0,500), N)
        X_class_0 = X_train[Y_train == -1]
        X_0 = X_class_0[rand_list]
        mu_estimate_0, sigma_estimate_0 = MLE_gaussian(X_0, N)

        rand_list = random.sample(range(0,500), N)
        X_class_1 = X_train[Y_train == 1]
        X_1 = X_class_1[rand_list]
        mu_estimate_1, sigma_estimate_1 = MLE_gaussian(X_1, N)

        theta_0 = [mu_estimate_0, sigma_estimate_0]
        theta_1 = [mu_estimate_1, sigma_estimate_1]

        Y_pred_bayes = bayes(theta_0, theta_1, X_test)
        Y_pred_bayes[Y_pred_bayes == 0] = -1
        Y_pred_bayes[Y_pred_bayes == 1] = 1

        Y_pred_nearest_neighbour = nearest_neighbour(X_0, X_1, X_test)

        bayes_accuracy = accuracy(Y_pred_bayes, Y_test)
        nearest_neighbour_accuracy = accuracy(Y_pred_nearest_neighbour, Y_test)

        output_file.write("{}, {}, {}\n".format(N, bayes_accuracy, nearest_neighbour_accuracy))

    output_file.close()


if __name__ == "__main__":
    problem_2a()
