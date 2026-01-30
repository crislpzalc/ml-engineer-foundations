
# Basic initialitation

def square (x):
    return x*x

def mean(values):
    sum = 0
    for i in range(len(values)-1):
        sum += values[i]
    return sum/len(values)

lista = [1, 2, 3, 4, 5, 6]
ave = mean(lista)
print(square(ave))

# Review zip and enumerate

labels = [0, 1, 0, 1, 1, 1]

for i, (x,y) in enumerate(zip(lista, labels)):
    print(f"Index {i}: value={x}, label={y}")

# Review dictionaries

experiment_results = {
    'acc': 0.99,
    'recall': 0.97,
    'f1': 0.9
}

for metric, value in experiment_results.items():
    print(metric, value)

# Review list comprehensions

# version 1
squared = []
for i in lista:
    squared.append(square(i))
print(squared)

# version 2
squared2 = [square(x) for x in lista]
print(squared2)

# numpy arrays
import numpy as np

a = np.array([1, 2, 3])
b = np.array([[1, 2, 3],
             [4, 5, 6]])

print(a)
print(b)

print(a.shape)
print(b.shape)

print(a*2)

c = np.array([10, 20, 30])
print(a+c)

# Review broadcasting

X = np.array([[10, 20, 30, 40, 50], 
             [60, 70, 85, 90, 100]])

mean_per_column = X.mean(axis=0)

mean_per_row = X.mean(axis=1)

print(mean_per_column, mean_per_row)

X_centered = X - mean_per_column

print(X_centered)

# Understand errors
# wrong = X + np.array([1, 2])
# print(wrong)

# Mini MSE

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

X = np.array([[1], [2], [3], [4]])
w = np.array([2.0])
b = 1.0

y_pred = X @ w + b
print(y_pred)
