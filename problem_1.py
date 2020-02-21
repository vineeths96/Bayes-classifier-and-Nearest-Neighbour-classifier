from problem_1.load_data import load_data
from problem_1.problem_1a import problem_1a
from problem_1.problem_1b import problem_1b

X_train, Y_train, X_test, Y_test = load_data('a')
problem_1a(X_train, Y_train, X_test, Y_test)

X_train, Y_train, X_test, Y_test = load_data('b')
problem_1b(X_train, Y_train, X_test, Y_test)