import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

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
mean_sq_err = metrics.mean_squared_error(y_test, prediction)

RMSE = np.sqrt(mean_sq_err)


# fewer features
X = house[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'yr_built', 'zipcode']]
y = house['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 7)
model2 = LinearRegression()
model2.fit(X_train, y_train)

prediction2 = model2.predict(X_test)
mean_sq_err2 = metrics.mean_squared_error(y_test, prediction2)

RMSE2 = np.sqrt(mean_sq_err2)

# RMSE가 RMSE2보다 더 적으므로 첫번째 모델이 더 효과적임을 알 수 있다.
# Coefficient 값을 토대로 절대값이 높은 항목이 영향을 크게 미친다.

# build another model and consider
X = house[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']]
y = house['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 7)
model3 = LinearRegression()
model3.fit(X_train, y_train)

prediction3 = model3.predict(X_test)
mean_sq_err3 = metrics.mean_squared_error(y_test, prediction3)

RMSE3 = np.sqrt(mean_sq_err3)
