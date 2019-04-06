import abc
import random

import numpy as np


class SGD:
    def __init__(self, model):
        self.model = model

    def __call__(self, X, y, epochs=30, batch_size=64, alpha=.01, _lambda=0):
        for epoch in range(epochs):
            print(f'\nepoch: {epoch}')
            print('-' * 79)
            # X = np.concatenate((np.ones((len(X), 1)), X), axis=1)
            batches = [(X[i:i + batch_size], y[i:i + batch_size]) for i in range(0, len(X), batch_size)]
            random.shuffle(batches)
            for b_X, b_y in batches:
                h = self.model.predict(b_X)
                error = self.model.error(h, b_y, _lambda)
                self.model.theta -= alpha * self._error_grad(h, b_X, b_y, self.model.theta, _lambda)
                print(f'error: {error} | theta: {self.model.theta}')

    def _error_grad(self, h, X, y, theta, _lambda):
        return 1/len(h) * np.dot((h-y), X) + 2*_lambda*theta


class Model(abc.ABC):
    optimizer = SGD
    def __init__(self, features=1):
        self.theta = np.ones(features + 1)
        self.optimizer = self.optimizer(self)

    def fit(self, X, y, **kwargs):
        self.optimizer(X, y, **kwargs)

    @abc.abstractmethod
    def error(self, h, y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass


class LinearRegressor(Model):
    def error(self, h, y, _lambda):
        return 1/(2*len(h)) * np.sum((h-y)**2) + _lambda*np.sum(self.theta)

    def normal(self, X, y):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        return np.inner(self.theta, X)


class LogisticRegressor(Model):
    def error(self, h, y):
        return -1/len(h) * np.sum(np.dot(y, np.log(h)) - np.dot((1-y), np.log(1-h)))

    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))

    def predict(self, X):
        return self.sigmoid(np.inner(self.theta, X))


X = np.random.rand(2000, 4)
X = np.concatenate((np.ones((len(X), 1)), X), axis=1)
y = (np.inner([1, 2, 3, 4, 5], X))
reg = LinearRegressor(
    features=4,
)
reg.fit(X, y, alpha=.25, batch_size=100)

print(f'\ntheta: {reg.theta}')
    # print(reg.normal(X, y))`
