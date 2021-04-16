#Santiago LE JEUNE ; FEDERICO AMATO
#HW2 helper functions, edits made to hw2_utils by Jake Vestal

from __future__ import print_function
from __future__ import absolute_import

from Strategy import *

import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from math import log, isnan

def quick_date_cols(df, colnames):
    for col in colnames:
        df[col] = pd.to_datetime(df[col]).dt.date
    return df

def to_years(x):
    str_split = x.lower().split()
    if len(str_split) == 2:
        if str_split[1] == 'mo':
            return int(str_split[0]) / 12
        if str_split[1] == 'yr':
            return int(str_split[0])

def Y_m_d_to_unix_str(ymd_str):
    return str(int(time.mktime(pd.to_datetime(ymd_str).date().timetuple())))

def fetch_usdt_rates(YYYY):
    # Requests the USDT's daily yield data for a given year. Results are
    #   returned as a DataFrame object with the 'Date' column formatted as a
    #   pandas datetime type.

    URL = 'https://www.treasury.gov/resource-center/data-chart-center/' + \
          'interest-rates/pages/TextView.aspx?data=yieldYear&year=' + str(YYYY)

    cmt_rates_page = requests.get(URL)

    soup = BeautifulSoup(cmt_rates_page.content, 'html.parser')

    table_html = soup.findAll('table', {'class': 't-chart'})

    df = pd.read_html(str(table_html))[0]
    df.Date = pd.to_datetime(df.Date)

    return df

def fetch_GSPC_data(start_date, end_date):
    # Return GSPC data

    URL = 'https://finance.yahoo.com/quote/%5EGSPC/history?' + \
            'period1=' + Y_m_d_to_unix_str(start_date) + \
            '&period2=' + Y_m_d_to_unix_str(end_date) + \
            '&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true'

    gspc_page = requests.get(URL)

    soup = BeautifulSoup(gspc_page.content, 'html.parser')

    table_html = soup.findAll('table', {'data-test': 'historical-prices'})


    df = pd.read_html(str(table_html))[0]

    df.drop(df.tail(1).index, inplace=True)

    # see formats here: https://www.w3schools.com/python/python_datetime.asp
    df.Date = pd.to_datetime(df.Date)

    cols = df.columns.drop('Date')
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce') #Convert to floats

    return df

def get_usdt_rates_last_n_days(cmt_rates, base_date, N):
    #cmt_rates is a df of cmt rates listed in date order
    # Base date is the last date in date range
    # N = # of days before Base_date to get data for

    #Assuming CMT rates alrd has average column

    base_date_index = cmt_rates.loc[cmt_rates['Date'].dt.date == base_date].index
    base_date_index = base_date_index[0]
    start_date_index = base_date_index - N

    cmt_rates_subset = cmt_rates.loc[start_date_index: base_date_index]

    return cmt_rates_subset


#########################################
#Backtest functions

def calc_features(bond_hist, ivv_hist, N, CI):
    # bond_hist is df of historical bond yields, including avg bond yield on any given day
    # ivv hist is df of historical data on ivv
    # N is number of days to look back on for linear regression on average yield
    # CI is confidence interval for UB and LB of prediction
    # Returns bond_features df: avg yield and ivv corr for last N days, yield volatility for last n days
    #                           a - linreg coef, b - linreg intercept, r2 of linreg, predicted avg yield for day N+1,
    #                           predicted UB & LB for day N+1
    bond_hist['Bond Avg'] = bond_hist.loc[:, '1 mo':'30 yr'].mean(axis=1)

    bond_and_ivv_hist = pd.merge(bond_hist, ivv_hist, on='Date')

    n = len(bond_and_ivv_hist['Date'])
    X = pd.DataFrame(np.arange(N))

    bond_features= np.zeros([n, 8])
    for i in range(n):
        dt = bond_and_ivv_hist['Date'][i]

        training_indices = bond_and_ivv_hist['Date'] < dt
        training_data = bond_and_ivv_hist[training_indices]

        if len(training_data['Date']) < N:
            bond_features[i] = [None, None, None, None, None, None, None, None]
            continue
        else:
            training_data = training_data.tail(N)  # trading data for last N days before trading_day


            bond_stock_corr = training_data['Bond Avg'].corr(training_data['Close'])

            linreg = LinearRegression()  # initialize linear regression model
            linreg = linreg.fit(X, training_data['Bond Avg'])

            mean_yield_last_n_days  = training_data['Bond Avg'].mean()
            std_yield_last_n_days   = np.std(training_data['Bond Avg'])
            bond_vol = std_yield_last_n_days/ mean_yield_last_n_days

            modeled_rates       =  linreg.predict(pd.DataFrame(np.arange(N+1)))
            r2                  =  r2_score(training_data['Bond Avg'], modeled_rates[:N])

            pred_LB, pred_UB    = stats.norm.interval(CI, loc = modeled_rates[-1], scale = std_yield_last_n_days)

            bond_features[i] = [bond_stock_corr, bond_vol, linreg.coef_[0],
                                linreg.intercept_, r2, modeled_rates[-1],
                                pred_LB, pred_UB]

    df = pd.DataFrame(bond_features)
    df.columns = ["Bond ivv corr last N", "Yield Vol last N", "a", "b", "R2", "Predicted avg yield", "Predicted yield LB", "Predicted yield UB"]

    df = pd.concat([bond_and_ivv_hist['Date'], df, bond_and_ivv_hist['Bond Avg']], axis = 1)

    return df

def calc_long_response(features, ivv_hist, n, alpha):

    response = []

    for features_dt in features['Date']:
        # Get data for the next n days after response_date
        ohlc_data_same_day = ivv_hist[['Date', 'Open', 'High', 'Low', 'Close']][
            ivv_hist['Date'] == features_dt]

        ohlc_data = ivv_hist[['Date', 'Open', 'High', 'Low', 'Close']][
            ivv_hist['Date'] > features_dt
            ].head(n)

        if len(ohlc_data) == 0:
            response_row = np.repeat(None, 8).tolist()
            response.append(response_row)
            continue

        entry_date   = features_dt
        entry_price  = ohlc_data_same_day['Close'].head(1).item()
        target_price = entry_price * (1 + alpha)

        # Find the earliest date and price in ohlc_data on which the high was
        # higher than target_price (meaning that the SELL limit order filled), if
        # they exist. If not, then return the date and closing price of the last
        # row in ohlc_data.
        (exit_date, exit_price) = next(
            (tuple(y) for x, y in
             ohlc_data[['Date', 'High']].iterrows() if
             y[1] >= target_price),
            (ohlc_data['Date'].values[-1], ohlc_data['Close'].values[-1])
        )

        highest_price = max(ohlc_data['High'])
        highest_price_date = ohlc_data['Date'][
            ohlc_data['High'] == highest_price
            ].values[0]

        success = int(exit_price >= target_price)

        if len(ohlc_data) < n and success == 0:
            exit_date = exit_price = highest_price = highest_price_date = \
                success = None

        response_row = [
            entry_date, entry_price, target_price, exit_date, exit_price,
            highest_price, highest_price_date, success
        ]

        response.append(response_row)

    response = pd.DataFrame(response)
    response.columns = [
        "long_entry_date", "long_entry_price", "long_target_price", "long_exit_date", "long_exit_price",
        "long_highest_price", "long_highest_price_date", "long_success"
    ]
    response = response.round(2)

    response.to_csv('app_data/long_response.csv', index=False)

    return response

def calc_short_response(features, ivv_hist, n, alpha):
    response = []

    for features_dt in features['Date']:
        # get data for same day
        ohlc_data_same_day = ivv_hist[['Date', 'Open', 'High', 'Low', 'Close']][
            ivv_hist['Date'] == features_dt]
        # Get data for the next n days after response_date
        ohlc_data = ivv_hist[['Date', 'Open', 'High', 'Low', 'Close']][
            ivv_hist['Date'] > features_dt
            ].head(n)

        if len(ohlc_data) == 0:
            response_row = np.repeat(None, 8).tolist()
            response.append(response_row)
            continue

        entry_date = features_dt
        entry_price =  ohlc_data_same_day['Close'].head(1).item()  #for short position, negative price
        target_price = entry_price * (1-alpha)                      # aim to close out at (1-alpha) entry_price
                                                                    # target price listed here as positive

        # Find the earliest date and price in ohlc_data on which the low was
        # lower than target_price (meaning that the BUY limit order filled), if
        # they exist. If not, then return the date and closing price of the last
        # row in ohlc_data.

        (exit_date, exit_price) = next(
            (tuple(y) for x, y in
             ohlc_data[['Date', 'Low']].iterrows() if
             y[1] <= target_price),
            (ohlc_data['Date'].values[-1], ohlc_data['Close'].values[-1])
        )

        lowest_price = min(ohlc_data['Low'])
        lowest_price_date = ohlc_data['Date'][
            ohlc_data['Low'] == lowest_price
            ].values[0]

        success = int(exit_price <= target_price)

        if len(ohlc_data) < n and success == 0:
            exit_date = exit_price = highest_price = highest_price_date = \
                success = None

        response_row = [
            entry_date, entry_price, target_price, exit_date, exit_price,
            lowest_price, lowest_price_date, success
        ]

        response.append(response_row)

    response = pd.DataFrame(response)
    response.columns = [
        "short_entry_date", "short_entry_price", "short_target_price", "short_exit_date", "short_exit_price",
        "short_lowest_price", "short_lowest_price_date", "short_success"
    ]
    response = response.round(2)

    response.to_csv('app_data/short_response.csv', index=False)

    return response

def calc_blotter(features_and_responses, start_date, end_date, rho, lot_size):
    blotter = []
    trade_id = 0

    for trading_date in features_and_responses['Date'][
        (features_and_responses['Date'] >= pd.to_datetime(start_date)) & (
                features_and_responses['Date'] <= pd.to_datetime(end_date)
        )
    ]:
        trade_decision = trading_decision(features_and_responses, trading_date, rho)

        if trade_decision['BUY'].item() == 1:
            right_answer = features_and_responses[
                features_and_responses['Date'] == trading_date
                ]

            # Create a market BUY order to enter.
            if trading_date == features_and_responses['Date'].tail(1).item():
                order_status = 'PENDING'
                submitted = order_price = fill_price = filled_or_cancelled = None
            else:
                submitted = filled_or_cancelled = right_answer[
                    'long_entry_date'].item()
                order_price = fill_price = right_answer['long_entry_price'].item()
                order_status = 'FILLED'

            entry_trade_mkt = [
                trade_id, 'L', submitted, 'BUY', lot_size, 'IVV',
                order_price, 'MKT', order_status, fill_price,
                filled_or_cancelled
            ]

            blotter.append(entry_trade_mkt)

            # Create the limit order to exit position.

            # Either successful (success == 1), unsuccessful (0), or it's
            # unsuccessful so far but n days haven't passed yet (NaN).
            success = right_answer['long_success'].item()

            # If unknown, then the limit order is still open and hasn't filled.
            if isnan(success):
                if trading_date == features_and_responses['Date'].tail(
                        1).item():
                    order_status = 'PENDING'
                else:
                    order_status = 'OPEN'
                order_price = right_answer['long_target_price'].item()
                fill_price = filled_or_cancelled = None

            # If limit order failed after n days:
            if success == 0:
                order_status = 'CANCELLED'
                order_price = right_answer['long_target_price'].item()
                fill_price = None
                filled_or_cancelled = right_answer['long_exit_date'].item()
                # Don't forget the market order to close position:
                exit_trade_mkt = [
                    trade_id, 'L', filled_or_cancelled, 'SELL', lot_size, 'IVV',
                    right_answer['long_exit_price'].item(), 'MKT', 'FILLED',
                    right_answer['long_exit_price'].item(), filled_or_cancelled
                ]
                blotter.append(exit_trade_mkt)

            # If the trade was successful:
            if success == 1:
                order_status = 'FILLED'
                order_price = right_answer['long_target_price'].item()
                fill_price = right_answer['long_exit_price'].item()
                filled_or_cancelled = right_answer['long_exit_date'].item()

            exit_trade_lmt = [
                trade_id, 'L', submitted, 'SELL', lot_size, 'IVV',
                order_price, 'LIMIT', order_status, fill_price,
                filled_or_cancelled
            ]

            blotter.append(exit_trade_lmt)
            trade_id += 1

        if trade_decision['SELL'].item() == 1:
            right_answer = features_and_responses[
                features_and_responses['Date'] == trading_date
                ]

            # Create a market order to enter.
            if trading_date == features_and_responses['Date'].tail(1).item():
                order_status = 'PENDING'
                submitted = order_price = fill_price = filled_or_cancelled = None
            else:
                submitted = filled_or_cancelled = right_answer[
                    'short_entry_date'].item()
                order_price = fill_price = right_answer['short_entry_price'].item()
                order_status = 'FILLED'

            entry_trade_mkt = [
                trade_id, 'S', submitted, 'SELL', lot_size, 'IVV',
                order_price, 'MKT', order_status, fill_price,
                filled_or_cancelled
            ]

            blotter.append(entry_trade_mkt)

            # Create the limit order to exit position.

            # Either successful (success == 1), unsuccessful (0), or it's
            # unsuccessful so far but n days haven't passed yet (NaN).
            success = right_answer['short_success'].item()

            # If unknown, then the limit order is still open and hasn't filled.
            if isnan(success):
                if trading_date == features_and_responses['Date'].tail(
                        1).item():
                    order_status = 'PENDING'
                else:
                    order_status = 'OPEN'
                order_price = right_answer['short_target_price'].item()
                fill_price = filled_or_cancelled = None

            # If limit order failed after n days:
            if success == 0:
                order_status = 'CANCELLED'
                order_price = right_answer['short_target_price'].item()
                fill_price = None
                filled_or_cancelled = right_answer['short_exit_date'].item()
                # Don't forget the market order to close position:
                exit_trade_mkt = [
                    trade_id, 'S', filled_or_cancelled, 'BUY', lot_size, 'IVV',
                    right_answer['short_exit_price'].item(), 'MKT', 'FILLED',
                    right_answer['short_exit_price'].item(), filled_or_cancelled
                ]
                blotter.append(exit_trade_mkt)

            # If the trade was successful:
            if success == 1:
                order_status = 'FILLED'
                order_price = right_answer['short_target_price'].item()
                fill_price = right_answer['short_exit_price'].item()
                filled_or_cancelled = right_answer['short_exit_date'].item()

            exit_trade_lmt = [
                trade_id, 'S', submitted, 'BUY', lot_size, 'IVV',
                order_price, 'LIMIT', order_status, fill_price,
                filled_or_cancelled
            ]

            blotter.append(exit_trade_lmt)
            trade_id += 1

    blotter = pd.DataFrame(blotter)
    blotter.columns = [
        'ID', 'ls', 'submitted', 'action', 'size', 'symbol', 'price', 'type',
        'status', 'fill_price', 'filled_or_cancelled'
    ]
    blotter = blotter[
        (blotter['submitted'] >= pd.to_datetime(start_date)) & (
                blotter['submitted'] <= pd.to_datetime(end_date)
        )
        ]
    blotter = blotter.round(2)
    blotter.sort_values(by='ID', inplace=True, ascending=False)
    blotter.reset_index()

    blotter.to_csv('app_data/blotter.csv', index=False)

    return blotter
