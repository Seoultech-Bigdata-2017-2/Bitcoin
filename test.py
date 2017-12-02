import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('bitcoin.csv',delimiter='|')

data['Date'] = data['Date'].astype('datetime64[ns]')

data = data[245:]
X = data.loc[:,['Volume','Google Trending']].values
y = data['Open'].values

reg = LinearRegression().fit(X, y)

prediction = reg.predict(X)

plt.semilogy(data.Date, data['Open'], label="Original")
plt.semilogy(data.Date, prediction, label="trained")
plt.legend()
plt.show()
