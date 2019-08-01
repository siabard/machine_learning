import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

house = pd.read_csv('home_data.csv')

X = house[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_lot15', 'yr_built', 'condition', 'zipcode']]
y = house['price']

X.shape
y.shape

# training, testing data set split : 75% for training, 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 7)

print(X_train.shape, y_train.shape)

print(X_test.shape)

# Instantiate Linear Regression Model
model = LinearRegression()

# Training Model
model.fit(X_train, y_train)

# Predict using test set
prediction  = model.predict(X_test)


# Evaluate
plt.figure(figsize=(10, 6))
plt.scatter(y_test, prediction, c="red")
plt.show()

# Coeff

model.coef_
dframe_coef = pd.DataFrame(model.coef_, X.columns, columns = ['Coeffic Value'])

model.intercept_

# RMSE
