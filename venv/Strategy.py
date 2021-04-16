from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from math import log, isnan
from statistics import stdev
from numpy import repeat
from datetime import date, datetime, timedelta

def trading_decision(features_and_responses, trading_date, rho):

    # trading_date is a dt object
    # features_and_responses is a pd df with with bond yields + avg bond yields, and IVV data and responses for trades
    # rho is max correleation allowed between stocks and bond for trade to take place

    # trading_decision returns   a dataframe with BUY and SELL columns

    features_and_responses_trading_day = features_and_responses[features_and_responses['Date'] == trading_date].head(1)
    if features_and_responses_trading_day['Bond ivv corr last N'].item() < rho:
        if features_and_responses_trading_day["Bond Avg"].item() <= features_and_responses_trading_day["Predicted yield LB"].item() :
            trade_decision = [[1, 0]]
        elif features_and_responses_trading_day["Bond Avg"].item() >= features_and_responses_trading_day["Predicted yield UB"].item() :
            trade_decision = [[0, 1]]
        else:
            trade_decision = [[0, 0]]
    else:
        trade_decision = [[0, 0]]


    trade_decision = pd.DataFrame(trade_decision)
    trade_decision.columns = ["BUY", "SELL"]
    return trade_decision