import numpy as np
import matplotlib.pyplot as plt

# Generate a synthetic dataset
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Convert y to binary labels
y_binary = (y > y.mean()).astype(int)

# Split the dataset into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y_binary[:split_index], y_binary[split_index:]

# Add bias term to feature matrix
X_train_bias = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_bias = np.c_[np.ones((len(X_test), 1)), X_test]

# Logistic Regression using Gradient Descent
learning_rate = 0.01
n_iterations = 1000

theta = np.random.randn(2, 1)

for iteration in range(n_iterations):
    logits = X_train_bias.dot(theta)
    predictions = 1 / (1 + np.exp(-logits))
    errors = predictions - y_train
    gradients = X_train_bias.T.dot(errors) / len(X_train)
    theta -= learning_rate * gradients

# Make predictions on the test set
logits_test = X_test_bias.dot(theta)
predictions_test = 1 / (1 + np.exp(-logits_test))
predicted_labels = (predictions_test >= 0.5).astype(int)

# Evaluate the model
accuracy = np.mean(predicted_labels == y_test)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Plot the decision boundary
plt.scatter(X_test, y_test, c=predicted_labels, cmap=plt.cm.Paired)
plt.xlabel('Feature')
plt.ylabel('Binary Label')
plt.title('Binary Classification with Logistic Regression')
plt.show()
