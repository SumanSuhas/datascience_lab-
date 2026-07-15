# Program 1: Implement KNN Algorithm on Iris Dataset and predict flower classes

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()                          # load iris dataset

X = iris.data                               # X = features (sepal/petal measurements)
y = iris.target                             # y = target class (flower type)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0     # 30% data used for testing, randomness of data is avoided, each run picks the same testing and training data
)

knn = KNeighborsClassifier(n_neighbors=3)   # n_neighbors = number of nearest points considered

knn.fit(X_train, y_train)                   # train KNN model

y_pred = knn.predict(X_test)                # predict classes for test data

print("Predicted Classes:")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)   # compare actual and predicted labels

print("Accuracy =", accuracy)