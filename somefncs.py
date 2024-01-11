# some functions that will be useful in the context of statistical arbitrage

import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
import yfinance as yf

def get_historical_Data(tickers):
    """This function returns a pd dataframe with all of the adjusted closing information"""
    data = pd.DataFrame()
    names = list()
    for i in tickers:
        data = pd.concat([data, pd.DataFrame(yf.download(i, start=datetime(2020, 8, 26), end=datetime(2022, 8, 26)).iloc[:,4])], axis = 1)
        names.append(i)
    data.columns = names
    return data

def get_intervalhistorical_Data(tickers, startyear, startmonth, endyear, endmonth):
    """This function returns a pd dataframe with all of the adjusted closing information"""
    data = pd.DataFrame()
    names = list()
    for i in tickers:
        data = pd.concat([data, pd.DataFrame(yf.download(i, start=datetime(startyear, startmonth, 1), end=datetime(endyear, endmonth, 1)).iloc[:,4])], axis = 1)
        names.append(i)
    data.columns = names
    return data