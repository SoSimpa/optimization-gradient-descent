"""Gradient Descent Optimization Methods
Implements and visualizes Batch, Stochastic, and Mini-batch Gradient Descent
for linear regression. Aligns with CMSE 382 Optimization Methods in Data Science.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

# Generate synthetic linear regression data
n = 200
X = np.random.uniform(0, 10, n)
y = 3.5 * X + 2.0 + np.random.normal(0, 2.5, n)
X = X.reshape(-1, 1)

# Add bias term
X_b = np.c_[np.ones((n, 1)), X]

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    return (1/(2*m)) * np.sum((predictions - y)**2)

def batch_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m = len(y)
    theta = np.random.randn(2, 1)
    cost_history = []
    
    for i in range(n_iterations):
        gradients = (1/m) * X.T @ (X @ theta - y.reshape(-1, 1))
        theta = theta - learning_rate * gradients
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history

def stochastic_gradient_descent(X, y, learning_rate=0.01, n_epochs=50):
    m = len(y)
    theta = np.random.randn(2, 1)
    cost_history = []
    
    for epoch in range(n_epochs):
        indices = np.random.permutation(m)
        for i in indices:
            xi = X[i:i+1]
            yi = y[i]
            gradient = xi.T @ (xi @ theta - yi)
            theta = theta - learning_rate * gradient
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history

def minibatch_gradient_descent(X, y, learning_rate=0.01, n_epochs=100, batch_size=32):
    m = len(y)
    theta = np.random.randn(2, 1)
    cost_history = []
    
    for epoch in range(n_epochs):
        indices = np.random.permutation(m)
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            batch_idx = indices[start:end]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            gradients = (1/len(batch_idx)) * X_batch.T @ (X_batch @ theta - y_batch.reshape(-1, 1))
            theta = theta - learning_rate * gradients
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history

# Run all three methods
theta_batch, cost_batch = batch_gradient_descent(X_b, y, learning_rate=0.01, n_iterations=500)
theta_sgd, cost_sgd = stochastic_gradient_descent(X_b, y, learning_rate=0.005, n_epochs=50)
theta_mb, cost_mb = minibatch_gradient_descent(X_b, y, learning_rate=0.01, n_epochs=80, batch_size=32)

print('=== Final Parameters (true: intercept=2.0, slope=3.5) ===')
print(f'Batch GD:     intercept={theta_batch[0,0]:.3f}, slope={theta_batch[1,0]:.3f}')
print(f'SGD:          intercept={theta_sgd[0,0]:.3f}, slope={theta_sgd[1,0]:.3f}')
print(f'Mini-batch:   intercept={theta_mb[0,0]:.3f}, slope={theta_mb[1,0]:.3f}')

# Cost convergence plot
plt.figure(figsize=(10, 5))
plt.plot(cost_batch, label='Batch GD')
plt.plot(cost_sgd, label='Stochastic GD')
plt.plot(cost_mb, label='Mini-batch GD')
plt.xlabel('Iteration / Epoch')
plt.ylabel('Cost (MSE/2)')
plt.title('Cost Convergence: Different Gradient Descent Variants')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cost_convergence.png', dpi=150)
plt.show()

# Fit visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data')
x_line = np.linspace(0, 10, 100).reshape(-1, 1)
x_line_b = np.c_[np.ones((100, 1)), x_line]
plt.plot(x_line, x_line_b @ theta_batch, 'r-', linewidth=2, label='Batch GD fit')
plt.plot(x_line, x_line_b @ theta_mb, 'g--', linewidth=2, label='Mini-batch fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Fits via Gradient Descent')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regression_fits.png', dpi=150)
plt.show()

print('\nOptimization examples complete. Check plots.')
