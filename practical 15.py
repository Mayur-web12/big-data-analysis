# Step 1: Import libraries

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Load dataset

data = load_iris()

X = data.data
y = data.target

# Step 3: Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Step 4: Choose classifier

model = DecisionTreeClassifier()

# Step 5: Train model

model.fit(X_train, y_train)

# Step 6: Prediction

y_pred = model.predict(X_test)

# Step 7: Evaluation

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:\n",
      confusion_matrix(y_test, y_pred))

# Diagram

plt.figure(figsize=(12,8))

plot_tree(model, filled=True)

plt.title("Decision Tree Classification")

plt.show()
