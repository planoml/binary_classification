import numpy as np
import matplotlib.pyplot as plt

# Generate random data for binary classification
np.random.seed(42)

# Number of data points for each class
num_samples = 100

# Class 0
class_0 = np.random.normal(loc=[2, 2], scale=[1, 1], size=(num_samples, 2))
labels_0 = np.zeros((num_samples, 1))

# Class 1
class_1 = np.random.normal(loc=[6, 6], scale=[1, 1], size=(num_samples, 2))
labels_1 = np.ones((num_samples, 1))

# Concatenate data from both classes
data = np.concatenate((class_0, class_1), axis=0)
labels = np.concatenate((labels_0, labels_1), axis=0)

# Shuffle the data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)

data = data[indices]
labels = labels[indices]

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * data.shape[0])

x_train, y_train = data[:split_index], labels[:split_index]
x_test, y_test = data[split_index:], labels[split_index:]

# Plot the generated data
plt.scatter(class_0[:, 0], class_0[:, 1], label='Class 0', marker='o')
plt.scatter(class_1[:, 0], class_1[:, 1], label='Class 1', marker='x')
plt.title('Generated Data for Binary Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Define a simple logistic regression model
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_epochs=1000):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros((num_features, 1))
        self.bias = 0

        for epoch in range(self.num_epochs):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)

            # Compute loss
            loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

            # Backward pass
            dz = predictions - y
            dw = np.dot(X.T, dz) / num_samples
            db = np.sum(dz) / num_samples

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print the loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        return (predictions >= 0.5).astype(int)

# Train the logistic regression model
model = LogisticRegression()
model.train(x_train, y_train)

# Make predictions on the test set
predictions = model.predict(x_test)

# Evaluate the model
accuracy = np.mean(predictions == y_test)
print(f'Test Accuracy: {accuracy}')
