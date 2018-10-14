import numpy as np
import matplotlib.pyplot as plt
import linreg

X = np.random.rand(2000, 4)
X = np.concatenate((np.ones((len(X), 1)), X), axis=1)
# y = np.inner([1, 2, 3, 4, 5], X)
y = (np.inner([-5, 1, 2, 3, 4], X) >= 0).astype(int)
reg = linreg.Classifier(
    features=4,
    batch_size=100,
    epochs=30,
    )
reg.sgd(X, y, alpha=.25)

print(f'\ntheta: {reg.theta}')
# print(reg.normal(X, y))