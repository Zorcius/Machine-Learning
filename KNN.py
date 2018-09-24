import pandas as pd
import numpy as np

#define column names
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
# load data
iris = pd.read_csv("./data/iris.txt", header=None, names=names)
print(iris.head(3))
label=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# scatter plot sepal_width vs sepal_length and petal_length vs petal_width
import matplotlib.pyplot as plt
'''
plt.subplot(1,2,1)
plt.scatter(iris['sepal_width'], iris['sepal_length'],
             cmap=plt.cm.get_cmap())
plt.xlabel('sepal_width')
plt.ylabel('sepal_length')

plt.subplot(1,2,2)
plt.scatter(iris['petal_width'], iris['petal_length'])
plt.xlabel('petal_width')
plt.ylabel('petal_length')
'''
#plt.tight_layout()
#plt.show()

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X = iris.ix[:, 0:4]
y = iris['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
'''knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, pred))

# using cross-validation to find optimal k
neighbors = list(range(1, 50, 2))
cv_score = []

# perform 10 fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,X_train, y_train, cv=10, scoring='accuracy')
    cv_score.append(scores.mean())

# misclassification
MSE = [1-x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print(optimal_k)
plt.plot(neighbors, MSE)
plt.xlabel('neighbors k')
plt.ylabel('MSE')
plt.show()'''
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
from collections import Counter
def predict(X_train, y_train, X_test, k):
    distances = []
    targets = []
    for i in range(len(X_train)):
        d = np.sqrt(np.sum((X_test - X_train[i,:])**2))
        distances.append([d,i])
    distances = sorted(distances)
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])

    # return most common targets
    # most_common返回的是[(x,y)]形式。第一个索引0取出列表中的第一个tuple
    # 第二个索引0取出tuple中的第一个元素
    return Counter(targets).most_common(1)[0][0]

def KNearestNeighbor(X_train,y_train,X_test,predictions,k):
    # check if k larger than len(X_train)
    if k > len(X_train):
        raise ValueError

    for i in range(len(X_test)):
        pred = predict(X_train, y_train, X_test[i, :], k)
        predictions.append(pred)

predictions = []
try:

    KNearestNeighbor(X_train, y_train, X_test, predictions, k=7)
    predictions = np.asarray(predictions)
    acc_score = accuracy_score(y_test, predictions)
    print('accuracy score:', acc_score)
except ValueError:
    print("no more neighbours than training samples!")