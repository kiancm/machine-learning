import numpy as np

import learner

class LogisticRegressor(learner.Learner):
    def error(self, h, y):
        return -1/len(h) * np.sum(np.dot(y, np.log(h)) - np.dot((1-y), np.log(1-h)))

    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))

    def predict(self, X):
        return self.sigmoid(np.inner(self.theta, X))


X = np.random.rand(2000, 4)
X = np.concatenate((np.ones((len(X), 1)), X), axis=1)
# y = np.inner([1, 2, 3, 4, 5], X)
y = (np.inner([-5, 1, 2, 3, 4], X) >= 0).astype(int)
reg = LogisticRegressor(
    features=4,
    batch_size=100,
    epochs=30,
)
reg.sgd(X, y, alpha=.25)

print(f'\ntheta: {reg.theta}')
# print(reg.normal(X, y))`