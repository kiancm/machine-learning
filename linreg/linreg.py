import abc
import random

import numpy as np

class Learner(abc.ABC):
    def __init__(self, batch_size=1, epochs=30, features=1):
        self.theta = np.ones(features + 1)
        self.batch_size = batch_size
        self.epochs = epochs

    def sgd(self, X, y, alpha=.01):
        # X = np.concatenate((np.ones((len(X), 1)), X), axis=1)
        for epoch in range(self.epochs):
            print(f'\nepoch: {epoch}')
            print('-' * 82)
            batches = [(X[i:i + self.batch_size], y[i:i + self.batch_size]) for i in range(0, len(X), self.batch_size)]
            random.shuffle(batches)
            for b_X, b_y in batches:
                h = self.predict(b_X)
                error = self.error(h, b_y)
                self.theta -= alpha * self._error_grad(h, b_X, b_y)
                print(f'error: {error} | theta: {self.theta}')

    def _error_grad(self, h, X, y):
        return 1/len(h) * np.dot((h-y), X)

    @abc.abstractmethod
    def error(self, h, y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass


class Regressor(Learner):
    def error(self, h, y):
        return 1/(2*len(h)) * np.sum((h-y)**2)

    def normal(self, X, y):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        return np.inner(self.theta, X)


class Classifier(Learner):
    def error(self, h, y):
        return -1/len(h) * np.sum(np.dot(y, np.log(h)) - np.dot((1-y), np.log(1-h)))

    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))

    def predict(self, X):
      2  return self.sigmoid(np.inner(self.theta, X))