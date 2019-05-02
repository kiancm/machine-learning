import abc
import random

import numpy as np


class Model(abc.ABC):
    def __init__(
        self, batch_size=64, epochs=30, features=1, _lambda=0
    ):
        self.theta = np.ones(features + 1)
        self.batch_size = batch_size
        self.epochs = epochs
        self._lambda = _lambda

    def fit(self, X, y, alpha=.01):
        for epoch in range(self.epochs):
            print(f"\nepoch: {epoch}")
            print("-" * 83)
            batches = [
                (X[i : i + self.batch_size], y[i : i + self.batch_size])
                for i in range(0, len(X), self.batch_size)
            ]
            random.shuffle(batches)
            for b_X, b_y in batches:
                error = self.error(b_X, b_y)
                self.theta -= alpha * self.gradient(b_X, b_y)
                print(f"error: {error:010.5} | theta: {self.theta}")

    @abc.abstractmethod
    def error(self, X, y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def gradient(self, X, y):
        pass


class LinearRegression(Model):
    def normal(self, X, y):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def error(self, X, y):
        h = self.predict(X)
        m = len(h)
        return 1 / (2 * m) * np.sum((h - y) ** 2) + self._lambda * np.sum(
            self.theta[1:] ** 2
        )

    def predict(self, X):
        return np.inner(self.theta, X)

    def gradient(self, X, y):
        h = self.predict(X)
        return 1 / len(h) * np.dot((h - y), X) + 2 * self._lambda * self.theta


class LogisticRegression(Model):
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        return self.sigmoid(np.inner(self.theta, X))

    def error(self, X, y):
        h = self.predict(X)
        m = len(h)
        return (
            -1 / m * np.sum(np.dot(y, np.log(h)) - np.dot((1 - y), np.log(1 - h)))
        )

    def gradient(self, X, y):
        h = self.predict(X)
        return 1 / len(h) * np.dot((h - y), X) + 2 * self._lambda * self.theta

X = np.random.uniform(-2, 2, (10000, 4))
X = np.concatenate((np.ones((len(X), 1)), X), axis=1)
y = (np.inner([1, 2, 3, 4, 5], X) >= 0).astype(int)
reg = LogisticRegression(features=4, batch_size=100, epochs=30, _lambda=0)
reg.fit(X, y, alpha=.1)

print(f"\ntheta: {reg.theta}")