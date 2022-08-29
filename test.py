import numpy as np
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
import yfinance as yf

from statsmodels.tsa.stattools import adfuller

def get_historical_Data(tickers):
    """This function returns a pd dataframe with all of the adjusted closing information"""
    data = pd.DataFrame()
    names = list()
    for i in tickers:
        data = pd.concat([data, pd.DataFrame(yf.download(i, start=datetime(2020, 8, 26), end=datetime(2022, 8, 26)).iloc[:,4])], axis = 1)
        names.append(i)
    data.columns = names
    return data

ticks = ["DPZ", "AAPL", "GOOG", "AMD", "GME", "SPY", "NFLX", "BA", "WMT","TWTR","GS","XOM","NKE","FE", "META","BRK-B", "MSFT"] #Name of company (Dominos pizza)
d = get_historical_Data(ticks)
# print(d.shape)
# Most Recent Data
# d.tail()

corr_matrix = d.corr()

adfuller_matrix = np.ones([d.shape[1],d.shape[1]])

for i in range(0, d.shape[1]):
    for k in range(0, d.shape[1]):
        if(i!=k):
            #series = pd.Series(d.iloc[:, i].values / d.iloc[:, k].values)
            ADF = adfuller(d.iloc[:, i].values / d.iloc[:, k].values)
            adfuller_matrix[i][k] = ADF[1]

# for i in ticks:
#     for k in ticks:
#         if(i!=k):
#             #series = pd.Series(d.iloc[:, i].values / d.iloc[:, k].values)
#             ADF = adfuller(d[i].values / d[k].values)
#             adfuller_matrix[i][k] = ADF[1]


# create heatmap to examine pairwise relation between stocks
import seaborn as sn
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5), dpi=100)
sn.heatmap(corr_matrix, annot = True, vmin=-1, vmax=1)
# plt.show()

# create a heatmap of adfuller p-values
plt.figure(figsize=(15, 5), dpi=100)
sn.heatmap(adfuller_matrix, annot = True, vmin=0, vmax=1, xticklabels=ticks, yticklabels=ticks)
# plt.show()

# a good choice would a high correlation pair with a stationary spread/ratio
# for instance pick Google/GoldmanSachs

# plot price value
plt.figure(figsize=(8, 6), dpi=200)
GS = d['GS'] # Goldman Sachs Group Inc Stock
GOOG = d['GOOG'] # Alphabet Inc Class C
plt.plot(GS, label = "Goldman Sachs")
plt.plot(GOOG, label = "Google")
plt.legend()
# plt.show()

# plot the spread
plt.figure(figsize=(8, 6), dpi=200)
plt.plot(GS - 3*GOOG, label = 'Spread (GS - 3 X GOOG)')
plt.legend()
plt.title("Spread between Goldman Sachs and 3 x Google")

# plot the spread
plt.figure(figsize=(8, 6), dpi=200)
plt.plot(GS / (3*GOOG), label = 'Spread (GS / (3 X GOOG))')
plt.legend()
plt.title("Ration Goldman Sachs and (3 x Google)")


from statsmodels.tsa.stattools import adfuller
# Compute the ADF test for Google and Goldman Sachs
# With all time series, you want to have stationary data otherwise our data will be very hard to predict.

GS_ADF = adfuller(GS)
print('P value for the Augmented Dickey-Fuller Test is', GS_ADF[1])
GOOG_ADF = adfuller(GOOG)
print('P value for the Augmented Dickey-Fuller Test is', GOOG_ADF[1])
Spread_ADF = adfuller(GS - GOOG)
print('P value for the Augmented Dickey-Fuller Test is', Spread_ADF[1])
Ratio_ADF = adfuller(GS / GOOG)
print('P value for the Augmented Dickey-Fuller Test is', Ratio_ADF[1])

ratio = GS/GOOG

# Also, we can take a look at the price ratios between the two time series.
figure(figsize=(8, 6), dpi=200)
plt.plot(ratio, label = 'Price Ratio (GS / GOOG))')
plt.axhline(ratio.mean(), color='red')
plt.legend()
plt.title("Price Ratio between GS and GOOG")

z_score = (ratio - ratio.mean())/ratio.std()

plt.plot(z_score)
plt.axhline(z_score.mean(), color = 'black')
