# Import libraries

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

# Load dataset

url = "https://stats.idre.ucla.edu/stat/data/binary.csv"

data = pd.read_csv(url)

# Variables

X = data[['gpa']]

y = data['gre']

# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Model

model = LinearRegression()

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Print output

print("Intercept:", model.intercept_)

print("Coefficient:", model.coef_)

# Plot Graph

plt.scatter(X_test, y_test,
            color='blue',
            label='Actual Data')

plt.plot(X_test, y_pred,
         color='red',
         linewidth=2,
         label='Regression Line')

plt.xlabel("GPA")

plt.ylabel("GRE")

plt.title("Multiple Regression (GPA vs GRE)")

plt.legend()

plt.show()
