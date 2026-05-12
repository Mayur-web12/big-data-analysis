# Import required libraries

import pandas as pd

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn import metrics

import matplotlib.pyplot as plt

# Load dataset

iris = load_iris()

X = iris.data

y = iris.target

# Split data into training and testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Create Decision Tree model

model = DecisionTreeClassifier()

# Train model

model.fit(X_train, y_train)

# Make predictions

y_pred = model.predict(X_test)

# Print Accuracy

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Predict new sample

sample = [[5.1, 3.5, 1.4, 0.2]]

print("Prediction for sample:", model.predict(sample))

# Plot Decision Tree

plt.figure(figsize=(12,8))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.title("Decision Tree Diagram")

plt.show()
