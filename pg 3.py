# Program 11: Implement EM Algorithm using Gaussian Mixture Model

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture

iris = load_iris()                          # load iris dataset

X = iris.data[:, :2]                        # use first 2 features for visualization

gmm = GaussianMixture(
    n_components=3,                         # number of Gaussian distributions
    random_state=42
)

gmm.fit(X)                                  # train GMM using EM algorithm

labels = gmm.predict(X)                     # assign cluster labels

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("EM Algorithm - Gaussian Mixture Model")

plt.show()