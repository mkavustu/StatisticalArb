import numpy as np
import somefncs as sf
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
import yfinance as yf
import seaborn as sn
import matplotlib.pyplot as plt

ticks = ["DPZ", "AAPL", "GOOG", "AMD", "GME", "SPY", "NFLX", "BA", "WMT","GS","XOM","NKE","FE", "META","BRK-B", "MSFT"]
#d = sf.get_historical_Data(ticks)
d = sf.get_intervalhistorical_Data(ticks, 2023, 1, 2023, 9)
corr_matrix = d.corr()

adfuller_matrix = np.ones([d.shape[1],d.shape[1]])

for i in range(0, d.shape[1]):
    for k in range(0, d.shape[1]):
        if(i!=k):
            #series = pd.Series(d.iloc[:, i].values / d.iloc[:, k].values)
            ADF = adfuller(d.iloc[:, i].values / d.iloc[:, k].values)
            adfuller_matrix[i][k] = ADF[1]
adfuller_df = pd.DataFrame(adfuller_matrix, ticks, ticks)

# create heatmap to examine pairwise relation between stocks
plt.figure(figsize=(15, 5), dpi=100)
plt.title("Correlation Factor")
sn.heatmap(corr_matrix, annot = True, vmin=-1, vmax=1)
#plt.show()

# create a heatmap of adfuller p-values for stockX/stockY ratio
# plt.figure(figsize=(15, 5), dpi=100)
#plt.title("P-value for ADFuller Test")
# sn.heatmap(adfuller_df_matrix, annot = True, vmin=0, vmax=1, xticklabels=ticks, yticklabels=ticks)
#plt.show()

# create a heatmap of adfuller dataframe p-values for stockX/stockY ratio
plt.figure(figsize=(15, 5), dpi=100)
plt.title("P-value for ADFuller Test")
sn.heatmap(adfuller_df, annot = True, vmin=0, vmax=1)
plt.show()

#find ticker pairs that are higly correlated and stationary
Y = corr_matrix[((np.abs(corr_matrix) > 0.8) & (corr_matrix < 1))].stack().index.tolist()
#Y = corr_matrix[((corr_matrix > 0.5) & (corr_matrix < 1))].stack().index.tolist()
#print(Y)
X = adfuller_df[(adfuller_df < 0.1)].stack().index.tolist()
#print(X)

print(set(X).intersection(Y))

with open("Output.txt", "w") as text_file:
    print(f"Purchase Amount: {(set(X).intersection(Y))}", file=text_file)