import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 1)  # Generate 100 random values as the input feature
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)  # Generate corresponding target values with some noise

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions
X_test = np.array([[0.913]])  # Input value for prediction
y_pred = model.predict(X_test)

# Plot the data points and the regression line
plt.scatter(X, y, label='Data Points')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.scatter(X_test, y_pred, color='green', marker='x', s=100, label='Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Simple Linear Regression')
plt.show()

print(f"Prediction for X_test={X_test}: {y_pred[0][0]}")