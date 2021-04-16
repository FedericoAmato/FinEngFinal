#Backtest for Santiago LE JEUNE and FEDERICO AMATO's trading strategy

#imports
from hw2_utils_SLJ_FA import *
from ledger import *

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

def backtest( ivv_hist, bonds_hist, n, N, alpha, rho, CI, lot_size, start_date, end_date, starting_cash):
    # Strategy_data is a dataframe containing both bond and ivv data
    # N is # of days to look back on when determining trade_decision
    # n is number of days before an order stays on the market before closing out
    # lot_size = # shares bought/sold short per trade
    # start_date & end_date are self explanatory
    # starting_cash

    # Convert JSON data to dataframes
    ivv_hist = quick_date_cols(pd.read_json(ivv_hist), ['Date'])
    bonds_hist = quick_date_cols(pd.read_json(bonds_hist), ['Date'])
    print('hist_data converted from json')

    features = calc_features(bonds_hist, ivv_hist,  N, CI)
    print('features calculated')

    response_long = calc_long_response(features, ivv_hist, n, alpha)
    print('response_long calculated')

    response_short = calc_short_response(features, ivv_hist, n, alpha)
    print('response_short calculated')
    response = pd.concat([response_long, response_short], axis = 1)

    features_and_responses = pd.concat([features, response], axis = 1)

    del features
    del response
    blotter = calc_blotter(features_and_responses,
                    start_date, end_date, rho, lot_size)
    print('Blotter done')

    calendar_ledger = calc_calendar_ledger(
        blotter, starting_cash, ivv_hist, start_date
    )
    print('calendar ledger done')

    trade_ledger = calc_trade_ledger(blotter, ivv_hist)
    print('trade ledger done')

    return features_and_responses, blotter, calendar_ledger, trade_ledger