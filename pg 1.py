import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris = load_iris()                          # load iris dataset
X = iris.data[:, :2]                        # first 2 features for 2D plotting
y = iris.target                             # flower classes

model = SVC(kernel='linear')                # linear kernel creates straight boundary
model.fit(X, y)                             # train SVM model

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(                       # create grid points (coordinates) used to draw decision boundaries
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])   # predict class for grid points
Z = Z.reshape(xx.shape)                            # reshape predictions into grid

plt.contourf(xx, yy, Z, alpha=0.3)         # plot decision regions
plt.scatter(X[:, 0], X[:, 1], c=y)         # plot actual samples

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("SVM Classification")

plt.show()