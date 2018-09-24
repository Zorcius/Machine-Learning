import numpy as np
from random import seed
from random import randrange
from csv import reader

# load csv file
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

#Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

#Convert string column to Integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

#Split dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = []
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds) # the size of each fold
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            idx = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(idx))
        dataset_split.append(fold)
    return dataset_split

#Calculate accuracy
def accuracy_metric(actual, pred):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == pred[i]:
            correct += 1
    return correct / float(len(actual)) * 100

#evaluate
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold) # maintain fold as test data
        train_set = sum(train_set, [])
        test_set = []
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

#make predictions with weights

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i+1] * row[i]
    return 1 if activation >= 0 else 0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, learning_rate, n_epochs):
    weights = [np.random.rand() for i in range(len(train[0]))]
    for epoch in range(n_epochs):
        sum_error = 0
        for row in train:
            pred = predict(row, weights)
            error = row[-1] - pred
            sum_error += error ** 2
            weights[0] = weights[0] + learning_rate * error
            for i in range(len(row) - 1):
                weights[i+1] += learning_rate * error * row[i]
        print('>epoch = %d, learning_rate = %.3f, error = %.3f' % (epoch, learning_rate, sum_error))
    return weights

#perceptron algorithm with stochastic gradient descent
def perceptron(train, test, l_rate, n_epoch):
    predictions = []
    weights = train_weights(train, l_rate, n_epoch)
    for row in test:
        pred = predict(row, weights)
        predictions.append(pred)
    return predictions

# test perceptron algorithm on sonar dataset
seed(1)

filename = './data/sonar.all-data.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i);

#convert string class to integers
str_column_to_int(dataset, len(dataset[0])-1)

n_folds = 3
l_rate = 0.01
n_epoch = 500
scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' %(sum(scores) / float(len(scores))))