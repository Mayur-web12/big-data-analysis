# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load dataset from web

url = "https://stats.idre.ucla.edu/stat/data/binary.csv"

data = pd.read_csv(url)

# Show dataset

print(data.head())

# Define variables

X = data[['gre', 'gpa', 'rank']]

y = data['admit']

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Create model

model = LogisticRegression(max_iter=200)

# Train model

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Accuracy

print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix

print("Confusion Matrix:\n",
      confusion_matrix(y_test, y_pred))

# Predict probabilities

prob = model.predict_proba(X)[:, 1]

# Plot graph

plt.scatter(data['gre'], prob,
            color='blue', label='Data')

plt.xlabel("GRE Score")

plt.ylabel("Probability of Admission")

plt.title("Logistic Regression Curve")

# Logistic Curve

sorted_gre = np.sort(data['gre'])

sorted_prob = prob[np.argsort(data['gre'])]

plt.plot(sorted_gre, sorted_prob,
         color='red',
         linewidth=2,
         label='Logistic Curve')

plt.legend()

plt.show()
