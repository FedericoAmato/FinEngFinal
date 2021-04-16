#Return data for the strategy
#Imports

from hw2_utils_SLJ_FA import *
import bloomberg_functions as bbg_f

def strategy_data(start_date, end_date, stock_bbg_identifier):
    # Take startDate and end_Date as datetime objects, return DataFrame with date required for strategy
    # DataFrame returned includes the average bond yield calculated as the mean of bond yields daily

    if end_date == date.today():
        end_date = date.today()-timedelta(days=1)

    start_year  = start_date.year
    end_year    = end_date.year
    years       = np.arange(start_year, end_year+1, 1)

    cmt_rates = fetch_usdt_rates(years[0])
    for i in range(len(years)-1):
        rates = fetch_usdt_rates(years[i+1])
        cmt_rates = pd.concat([cmt_rates, rates], axis=0)

    cmt_rates = cmt_rates.sort_values('Date')
    cmt_rates['Bond Avg'] = cmt_rates.loc[:, '1 mo':'30 yr'].mean(axis=1)
    cmt_rates = cmt_rates.set_index(np.arange(len(cmt_rates['Date'])), drop=True)

    start_date_index    = cmt_rates.loc[cmt_rates['Date'].dt.date == start_date].index[0]
    end_date_index      = cmt_rates.loc[cmt_rates['Date'].dt.date == end_date].index[0]
    cmt_rates           = cmt_rates.iloc[start_date_index: (end_date_index+1)]

    IVV_data = bbg_f.req_historical_data(stock_bbg_identifier, start_date, end_date)
    IVV_data.Date = pd.to_datetime(IVV_data.Date)

    data = pd.merge(cmt_rates, IVV_data, on='Date')

    return data