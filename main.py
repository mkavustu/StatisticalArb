#   Built on the initial work by Spencer Pao

import numpy as np
import somefncs as sf

from statsmodels.tsa.stattools import adfuller

ticks = ["DPZ", "AAPL", "GOOG", "AMD", "GME", "SPY", "NFLX", "BA", "WMT","GS","XOM","NKE","FE", "META","BRK-B", "MSFT"]
d = sf.get_historical_Data(ticks)
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

# create a heatmap of adfuller p-values for stockX/stockY ratio
plt.figure(figsize=(15, 5), dpi=100)
sn.heatmap(adfuller_matrix, annot = True, vmin=0, vmax=1, xticklabels=ticks, yticklabels=ticks)
# plt.show()

# a good choice would be a high correlation pair with a stationary spread/ratio
# for instance pick Google/GoldmanSachs (planning to add some interaction here later)

# plot price value
plt.figure(figsize=(8, 6), dpi=200)
GS = d['GS'] # Goldman Sachs Group Inc Stock
GOOG = d['GOOG'] # Alphabet Inc Class C
plt.plot(GS, label = "Goldman Sachs")
plt.plot(GOOG, label = "Google")
plt.legend()
plt.show()

# plot the spread
plt.figure(figsize=(8, 6), dpi=200)
plt.plot(GS - GOOG, label = 'Spread GS - GOOG')
plt.legend()
plt.title("Spread between Goldman Sachs and Google")

# plot the spread
plt.figure(figsize=(8, 6), dpi=200)
plt.plot(GS / GOOG, label = 'Spread (GS / GOOG)')
plt.legend()
plt.title("Ratio of Goldman Sachs and Google")


from statsmodels.tsa.stattools import adfuller
# Compute the ADF test for Google and Goldman Sachs

GS_ADF = adfuller(GS)
print('P value for the Augmented Dickey-Fuller Test is', GS_ADF[1])
GOOG_ADF = adfuller(GOOG)
print('P value for the Augmented Dickey-Fuller Test is', GOOG_ADF[1])
Spread_ADF = adfuller(GS - GOOG)
print('P value for the Augmented Dickey-Fuller Test is', Spread_ADF[1])
Ratio_ADF = adfuller(GS / GOOG)
print('P value for the Augmented Dickey-Fuller Test is', Ratio_ADF[1])

# Interesting to note(or maybe not so): Even though GS-Google spread is non-stationary, it is possible to
#   get a stationary spread by looking at GS - 2.7*GOOG. Naturally, the question is, does an 'n' exist for
#   any spread such that spread of X and n*Y is stationary?

# Or more generally, does a non-trivial f exist for every X & Y such that f(X,Y) is non-stationary?
ratio = GS/GOOG
plt.figure(figsize=(8, 6), dpi=200)
plt.plot(ratio, label = 'Price Ratio (GS / GOOG))')
plt.axhline(ratio.mean(), color='black')
plt.legend()
plt.title("Price Ratio between GS and GOOG")


# normalize the ratio by its standard deviation
z_score = (ratio - ratio.mean())/ratio.std()
plt.plot(z_score)
plt.axhline(z_score.mean(), color='black', label='mean')
plt.axhline(1, color='red', label='Z score: 1')
plt.axhline(-1, color='green', label='Z score: -1')
plt.legend()
plt.title('Z score of GS/GOOG')



plt.figure(figsize=(8, 6), dpi=200)
# here a low pass filter, but any causal filter can be chosen depending on the application
ratios_mavg5 = ratio.rolling(window=5, center=False).mean()
# looks at the average ratio of the past n=5 days
ratios_mavg20 = ratio.rolling(window=20, center=False).mean()
# standard deviation of the ratio of past n=20 days
std_20 = ratio.rolling(window=20, center=False).std()

zscore_20_5 = (ratios_mavg5 - ratios_mavg20)/std_20
plt.plot(ratio.index, ratio.values)
plt.plot(ratios_mavg5.index, ratios_mavg5.values)
plt.plot(ratios_mavg20.index, ratios_mavg20.values)
plt.legend(['Ratio', '5d Ratio MA', '20d Ratio MA'])
plt.xlabel('Date')
plt.ylabel('Ratio')
plt.show()


plt.figure(figsize=(8, 6), dpi=200)
zscore_20_5.plot()
plt.axhline(0, color='black')
plt.axhline(1, color='red', linestyle='--')
plt.axhline(1.25, color='red', linestyle='--')
plt.axhline(-1, color='green', linestyle='--')
plt.axhline(-1.25, color='green', linestyle='--')
plt.legend(['Rolling Ratio z-score', 'Mean', '+1','+1.25','-1','-1.25'])
plt.show()


plt.figure(figsize=(8, 6), dpi=200)
ratio.plot()
buy = ratio.copy()
sell = ratio.copy()
buy[zscore_20_5>-1] = 0
sell[zscore_20_5<1] = 0
buy.plot(color='g', linestyle='None', marker='^')
sell.plot(color='r', linestyle='None', marker='^')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, 2, 3.5))
plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
plt.title('Relationship GS to GOOG')
plt.show()