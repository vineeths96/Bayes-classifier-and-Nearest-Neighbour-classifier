from sklearn.mixture import BayesianGaussianMixture

def EM_gaussian(X_train):
    bgsm = BayesianGaussianMixture(n_components=2)

    labels = bgsm.fit_predict(X_train)

    if (labels[0] == 0):
        class_0 = 0
        class_1 = 1
    else:
        class_0 = 1
        class_1 = 0

    means = bgsm.means_
    weights = bgsm.weights_
    covariances = bgsm.covariances_

    theta_0 = [means[0], covariances[0]]
    theta_1 = [means[1], covariances[1]]

    return [theta_0, theta_1], weights


if __name__ == "__main__":
    EM_gaussian()