# Setting up our data and visualize our data
# Univariate Linear Regression (Using Numpy)
# Use scikit learn to implment a multivar regression

# https://spark-public.s3.amazonaws.com/dataanalysis/loansdata.csv

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

loans_data = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

# loans_data.head()
# loans_data['Loan.Length'] --> 불필요한 Month 같은 내역을 빼야함.
# loans_data['Interest.Rate'][0:10] --> % 기호등은 필요없다.
# loans_data['FICO.Range'][0:10] --> 대시(-)로 구분된 값을 처리해야한다.
# "715-719" 로 된 값을 (715, 719)로 변환해야함

# Cleansing data

loans = pd.read_csv('load_csv')
loans.head()

# Histogram으로 데이터의 양태를 추적한다.
plt.figure()

fico = loans['FICO.Score']
fico.hist(bins = 20)

# Box Plot
plt.figure()
x = loans.boxplot('Interest.Rate', 'FICO.Score')
x.set_xticklabels(['630',  '', '', '', '', '660',  '', '', '', '', '690',  '', '', '', '', '720',  '', '', '', '', '750',  '', '', '', '', '780',  '', '', '', '', '810',  '', '', '', '', '840'])
x.set_xlabel("FICO Score")
x.set_ylabel("Interest Rate in %")
x.set_title("BOX PLOT")

plt.show()

# Scatterplot Matrix
pd.plotting.scatter_matrix(loans, alpha = 0.1, figsize=(10, 10), diagonal = 'hist')
