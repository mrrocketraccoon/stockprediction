import pandas as pd
import datetime
import pandas_datareader.data as web
import math
import numpy as np
from pandas import Series, DataFrame
from sklearn import preprocessing
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

df = pd.read_csv("BMW.DE.csv", index_col=0)
#Feature extraction: HL_PCT =High Low Percentage, PCT = Percentage Change
dfreg = df.loc[:,["Adj Close","Volume"]]
dfreg["HL_PCT"] = (df["High"] - df["Low"]) / df["Close"] * 100.0
dfreg["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0
#Data Preprocessing
# Drop missing value
dfreg.fillna(value=-99999, inplace=True)
# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.08 * len(dfreg)))
# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))
# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)
# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Model Generation
# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)
# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

#Bayesian Regression
clfbay = BayesianRidge(compute_score=True)
clfbay.fit(X_train, y_train)

#Evaluating models
confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test,y_test)
confidenceknn = clfknn.score(X_test, y_test)
confidencebay = clfbay.score(X_test, y_test)

#Predicting some stocks
forecast_set_linreg = clfreg.predict(X_lately)
#Predicting some stocks
forecast_set_knn = clfknn.predict(X_lately)
#Predicting some stocks
forecast_set_bayes = clfbay.predict(X_lately)
forecast_sets = {}
forecast_sets.update([('ForecastLinReg', forecast_set_linreg), ('ForecastKNN', forecast_set_knn), ('ForecastBayes',forecast_set_bayes)])

#Plotting predictions
import matplotlib.pyplot as plt
from matplotlib import style

# Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

last_date = dfreg.iloc[-1].name
last_unix = last_date
last_unix = datetime.datetime.strptime(last_date, '%Y-%m-%d')

for forecast_set in forecast_sets:
	next_unix = last_unix + datetime.timedelta(days=1)
	dfreg[forecast_set] = np.nan
	forecast_set_values = forecast_sets.get(forecast_set)
	for i in forecast_set_values:
	    next_date = next_unix
	    next_unix += datetime.timedelta(days=1)
	    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
	dfreg['Adj Close'].tail(500).plot()
	dfreg[forecast_set].tail(500).plot()
	plt.legend(loc=4)
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.show()