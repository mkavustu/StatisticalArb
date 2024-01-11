import numpy as np
import somefncs as sf
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
import yfinance as yf
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ticks = ["AMD", "MSFT"]
d = sf.get_intervalhistorical_Data(ticks, 2023, 1, 2023, 9)

pair1 = d[ticks[0]] # pair1
pair2 = d[ticks[1]] # pair2
ratio = pair1/pair2

# plt.figure(figsize=(8, 6), dpi=200)
# plt.plot(ratio, label = 'Price Ratio (' + ticks [0] + ' / '+ ticks[1] + '))')
# plt.axhline(ratio.mean(), color='black')
# plt.legend()
# plt.title("Price Ratio between "+ ticks [0] + ' and '+ ticks[1])
#plt.show()

z_score = (ratio - ratio.mean())/ratio.std()
plt.figure(figsize=(8, 6), dpi=200)
plt.plot(z_score)
plt.axhline(z_score.mean(), color='black', label='mean')
plt.axhline(2, color='red', label='Z score: 2', linestyle = '--')
plt.axhline(-2, color='green', label='Z score: -2', linestyle = ':')
#plt.legend()
plt.title('Training Test ' + ticks [0] + ' / '+ ticks[1])

buy = z_score[z_score < -2].copy()
sell = z_score[z_score > 2].copy()

buy.plot(color='g', linestyle='None', marker= 6)
sell.plot(color='r', linestyle='None', marker= 7)

mean = mlines.Line2D([], [], color='black', marker= 'None', linestyle='-',
                          markersize=10, label='Mean')
red_arrow = mlines.Line2D([], [], color='red', marker= 'v', linestyle='None',
                          markersize=5, label='Sell Signal')
green_arrow = mlines.Line2D([], [], color='green', marker= '^', linestyle='None',
                          markersize=5, label='Buy Signal')
red_dashed = mlines.Line2D([], [], color='red', marker= 'None', linestyle='--',
                          markersize=5, label='Z score: 2')
green_star = mlines.Line2D([], [], color='green', marker= 'None', linestyle=':',
                          markersize=5, label='Z score: -2')

legendhandle = [mean, red_dashed, green_star, red_arrow, green_arrow]

plt.legend(handles=legendhandle)

plt.show()

backtest = sf.get_intervalhistorical_Data(ticks, 2023, 9, 2023, 12)

backtestpair1 = backtest[ticks[0]]
backtestpair2 = backtest[ticks[1]]
backtestratio = backtestpair1/backtestpair2

backtest_z_score = (backtestratio - ratio.mean())/ratio.std()
plt.figure(figsize=(8, 6), dpi=200)
plt.plot(backtest_z_score)
plt.axhline(z_score.mean(), color='black', label='mean')
plt.axhline(2, color='red', linestyle='--')
plt.axhline(-2, color='green', linestyle=':')
plt.legend()
plt.title('Backtest Z score of ' + ticks [0] + ' / '+ ticks[1])


buy = backtest_z_score[backtest_z_score < -2].copy()
sell = backtest_z_score[backtest_z_score > 2].copy()

buy.plot(color='g', linestyle='None', marker= 6)
sell.plot(color='r', linestyle='None', marker= 7)

mean = mlines.Line2D([], [], color='black', marker= 'None', linestyle='-',
                          markersize=10, label='Mean')
red_arrow = mlines.Line2D([], [], color='red', marker= 'v', linestyle='None',
                          markersize=5, label='Sell Signal')
green_arrow = mlines.Line2D([], [], color='green', marker= '^', linestyle='None',
                          markersize=5, label='Buy Signal')
red_dashed = mlines.Line2D([], [], color='red', marker= 'None', linestyle='--',
                          markersize=5, label='Z score: 2')
green_star = mlines.Line2D([], [], color='green', marker= 'None', linestyle=':',
                          markersize=5, label='Z score: -2')

legendhandle = [mean, red_dashed, green_star, red_arrow, green_arrow]

plt.legend(handles=legendhandle)
#plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
plt.title(ticks[0] + ' / '+ ticks[1])
plt.show()

# plt.figure(figsize=(8, 6), dpi=200)
# backtestratio.plot()
# buy = backtestratio.copy()
# sell = backtestratio.copy()
# buy[(backtestratio - ratio.mean())/ratio.std() > -1] = 0
# sell[(backtestratio - ratio.mean())/ratio.std() < 1] = 0
# buy.plot(color='g', linestyle='None', marker='^')
# sell.plot(color='r', linestyle='None', marker='^')
# #x1, x2, y1, y2 = plt.axis()
# #plt.axis((x1, x2, 2, 3.5))
# plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
# plt.title(ticks[0] + ' / '+ ticks[1])
# plt.show()