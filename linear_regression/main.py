# minimize sum of squre error (Mean Square Error)
# Y - Y' = Y - theta * Xt
# theta is vector
# X is parameter

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# loading dataset using pandas

house = pd.read_csv('home_data.csv')

# print head rows (default 5)
house.head()

# print tail rows
house.tail()

# 관련 DataFrame의 정보를 가져오기
house.info()

# 관련 DataFrame에 대한 설명
house.describe()

# DataFrame의 전체 컬럼
house.columns

# Y 값은 price, Feature는 bedrooms, bathrooms 등이 될 것임
# 해당 feature를 가지고 Y값을 추정(Predict)할 것임

# 스캐터그램을 출력해본다.
plt.figure(figsize=(10, 6))
plt.scatter(house.sqrt_living, house.price)
plt.xlabel('sqrt of house')
plt.ylabel('price of house')
plt.show()


# Linear Regression Line
sns.lmplot('sqft_living', 'price', data = house)
sns.heatmap(house.corr())
sns.distplot(house['price'], color = 'red')

# Visualize dataset
sns.boxplot(x='zipcode', y='price', data = house)
