# Program 12: Implement PCA on Breast Cancer Dataset

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_breast_cancer()                 # load breast cancer dataset

X = data.data                               # input features
y = data.target                             # target classes

scaler = StandardScaler()                   # standardize feature values
X_scaled = scaler.fit_transform(X)          # mean = 0 and standard deviation = 1

pca = PCA(n_components=2)                   # reduce data to 2 principal components

X_pca = pca.fit_transform(X_scaled)         # transform data into PCA space

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Breast Cancer Dataset")

plt.show()