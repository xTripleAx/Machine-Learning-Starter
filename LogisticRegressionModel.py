import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#Two Features
# Generate synthetic data for a binary classification task
np.random.seed(0)
X = np.random.randn(100, 3)  # Two features
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Binary target variable (0 or 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression(random_state=0)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# Scatter plot of the data points
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0', c='red', marker='o')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', c='blue', marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Scatter Plot of Binary Classification Data')
plt.show()




#One Feature
# Generate synthetic data for a binary classification task
np.random.seed(0)
X = np.random.randn(100, 1)  # Single feature
y = (X[:, 0] > 0).astype(int)  # Binary target variable (0 or 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression(random_state=0)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# Scatter plot of the data points
plt.scatter(X[y == 0], y[y == 0], label='Class 0', c='red', marker='o')
plt.scatter(X[y == 1], y[y == 1], label='Class 1', c='blue', marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Class')
plt.legend()
plt.title('Scatter Plot of Binary Classification Data')
plt.show()

# Plot the decision boundary
plt.figure()
plt.scatter(X[y == 0], y[y == 0], label='Class 0', c='red', marker='o')
plt.scatter(X[y == 1], y[y == 1], label='Class 1', c='blue', marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Class')
plt.legend()

# Plot the decision boundary
X_decision_boundary = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
y_decision_boundary = model.predict(X_decision_boundary)
plt.plot(X_decision_boundary, y_decision_boundary, color='green', linewidth=2, label='Decision Boundary')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()