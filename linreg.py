import numpy as np

from . import learner


class LinearRegressor(learner.Learner):
    def error(self, h, y):
        return 1 / (2 * len(h)) * np.sum((h - y) ** 2) + self._lambda * np.sum(
            self.theta[1:] ** 2
        )

    def normal(self, X, y):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        return np.inner(self.theta, X)


X = np.random.rand(2000, 4)
X = np.concatenate((np.ones((len(X), 1)), X), axis=1)
y = np.inner([1, 2, 3, 4, 5], X)
reg = LinearRegressor(features=4, batch_size=100, epochs=30, _lambda=0)
reg.sgd(X, y, alpha=0.25)

print(f"\ntheta: {reg.theta}")
# print(reg.normal(X, y))`

