import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate input data with a sinusoidal function
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Add Gaussian noise with mean 0 and variance 0.25
y_noisy = y + np.random.normal(loc=0, scale=0.25, size=100)

# Reshape the data for the regression model
x = x.reshape(-1, 1)

# Fit the regression model to the noisy data
reg = LinearRegression().fit(x, y_noisy)

# Predict the output using the trained model
y_pred = reg.predict(x)

# Plot the input data, noisy data, and the prediction
plt.plot(x, y, label="True Function (Linear Regression)")
plt.scatter(x, y_noisy, label="Noisy Data")
plt.plot(x, y_pred, label="Prediction")
plt.legend()
plt.show()

# Create a list to store the polynomial models
models = []

# Generate the polynomial features for all degrees from 1 to 9
for degree in range(1, 4):
    # Generate the polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(x).reshape(-1, 2)
    
    # Fit the regression model to the noisy data
    reg = LinearRegression().fit(X_poly, y_noisy)

    y_pred = reg.predict(x)

    # Plot the input data, noisy data, and the prediction
    plt.plot(x, y, label="True Function (Linear Regression)")
    plt.scatter(x, y_noisy, label="Noisy Data")
    plt.plot(x, y_pred, label="Prediction")
    plt.legend()
    plt.show()
    
   