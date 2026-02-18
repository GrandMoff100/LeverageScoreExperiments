import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 10, 100)
y = 2 * x + 1 + 2 * np.random.randn(100)

# y[2] = 100  # Add an outlier

# Regression
X = np.vstack([x, np.ones_like(x)]).T
beta = np.linalg.lstsq(X, y, rcond=None)[0]
plt.scatter(x, y, label="Data")
plt.plot(x, X @ beta, color="red", label="Fit")
# plt.plot(x[2], y[2], "ro", label="Outlier", color="orange")
plt.legend()
plt.savefig("regression.png")


def leverage_scores(X):
    Q, _ = np.linalg.qr(X)
    return np.sum(Q**2, axis=1)


plt.figure()
scores = leverage_scores(X)
plt.plot(x, scores, label="Leverage Scores")
# plt.plot(x[2], scores[2], "ro", label="Outlier")
plt.legend()
plt.savefig("leverage_scores.png")


X = np.array(
    [
        [1, 0],
        [0, 1],
        [1, 10],
    ]
)

print(leverage_scores(X))
