from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
from tqdm import tqdm



def aggregate_market_data():
    """
    Aggregates market data for all tickers within a specified period.

    Args:
        period_start (date): The start date for data aggregation.
        period_end (date): The end date for data aggregation.

    Returns:
        None: Writes the aggregated data to CSV files.
    """
    all_paths = os.listdir('data')
    all_df = []
    for ticker in tqdm(all_paths):
        try:
            mkt_data_ticker = pd.read_csv(os.path.join('data',ticker),index_col=0)
            mkt_data_ticker.columns = [ticker.replace('.csv',"")]
            all_df.append(mkt_data_ticker)
        except:
            print("failed {0}".format(ticker))
    return all_df


def prepare_data(data, from_year = "2015", start_year_test="2020"):
    df = pd.concat(data, axis=1).dropna(axis=1, how='any').pct_change().dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().loc[from_year:]
    sc = StandardScaler()
    df = pd.DataFrame(sc.fit_transform(df), index=df.index, columns=df.columns)
    df = pd.DataFrame(df.stack(), columns=['Ret'])
    df.index.names = ['date', 'Ticker']
    df_orig = df.copy()
    df.loc[start_year_test:] = np.nan
    df = df.swaplevel(0, 1)
    df_orig = df_orig.swaplevel(0, 1)
    df = df['Ret'].unstack().T

    start_date = df.index.min()
    end_date = df.index.max()
    new_index = pd.date_range(start_date, end_date, freq='B')
    df_reindexed = df.reindex(new_index)
    df_reindexed.index.names = ['date']

    return df_reindexed, df_orig, df