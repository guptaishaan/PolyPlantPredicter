import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('/Users/ishaangupta/Downloads/main/assderp/ssrn/10_4231_R72F7KK2/Agro-Climatic Data by County/final_data2011.csv')

# Extract the predictor variables (crop moisture and crop year) and target variable (crop yield)
X = data[['sPh', 'ppt']].values
y = data['yield'].values

# Split the data into training and testing sets (90% training, 10% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=1)

# Create polynomial features for quadratic regression
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Train the regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Make predictions on training and testing data
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# Calculate the accuracy of the model
train_accuracy = model.score(X_train_poly, y_train)
test_accuracy = model.score(X_test_poly, y_test)
test_accuracy = abs(test_accuracy)


print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)

# Plotting the model and data points
plt.scatter(X_train[:, 0], y_train, color='blue', label='Training Data')
plt.scatter(X_test[:, 0], y_test, color='red', label='Testing Data')

# Sort the data points by crop year for smoother curve visualization
sort_indices = np.argsort(X[:, 1])
X_sorted = X[sort_indices]
y_pred_sorted = model.predict(poly_features.transform(X_sorted))

plt.plot(X_sorted[:, 0], y_pred_sorted, color='green', label='Polynomial Regression Model')
plt.xlabel('Crop Moisture')
plt.ylabel('Crop Yield')
plt.legend()
plt.show()
