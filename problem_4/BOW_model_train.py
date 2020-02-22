from sklearn.naive_bayes import MultinomialNB


def model_train(X_train, Y_train):
    model = MultinomialNB()

    model.fit(X_train, Y_train)

    return model


if __name__ == "__main__":
    model_train()