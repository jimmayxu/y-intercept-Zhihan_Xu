import numpy as np
import pandas as pd


def filtering(df, sample_thres = 500):
    """
    Remove stock if the sample is less than 500.
    :param df: data frame
    :param sample_thres: threshold for number of samples
    :return: filtered data frame
    """
    counts = df.ticker.value_counts()
    remaining = counts.index[counts >= sample_thres]
    df_ = df[df.ticker.isin(remaining)]
    df_ = df_.reset_index()
    df_ = df_.set_index('index')
    return df_



def price_ratio(df, insert_nan = 50):
    """
    :param df: data frame of close price
    :param insert_nan: this number matches with the rolling number in feature enginnering
    :return: price_after / price_before - 1, while first [insert_nan] of entries are set to nan
    """
    last = df['last']
    ratio = last[1:].to_numpy() / last[:-1].to_numpy() - 1
    ratio = np.append(np.nan, ratio)
    ratio[:insert_nan] = np.nan
    return pd.Series(ratio)


def feature_enginnering(df, rolling = 50):
    """
    Computation of quantitative indicators in a rolling of 50.

    Considering only close price, those include
    Simple Moving Average (SMA), Exponential Moving Average (EMA) and
    Relative Strength Index(RSI).

    Considering volume only, those include
    Volume Rate of Change (VROC).

    Considering close price and volume, those include
    On-Balance Volume (OBV), Volume Weighted Moving Average (VWMA), Force Index (FI)

    :param df: data frame including close price and volume
    :return: table of 6 factors in total
    """
    last = df['last']
    vol = df['volume']
    feature_table = pd.DataFrame()

    # compute SMA
    feature_table['SMA'] = last.rolling(window=rolling).mean()
    # compute EMA
    feature_table['EMA'] = last.ewm(span=rolling, adjust=False).mean()

    # Compute RSI
    delta = last.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    average_gain = gain.rolling(window=rolling).mean()
    average_loss = loss.rolling(window=rolling).mean()
    relative_strength = average_gain / average_loss
    feature_table['RSI'] = 100 - (100 / (1 + relative_strength))

    # Compute VROC
    feature_table['VROC'] = (vol / vol.shift(rolling) - 1) * 100

    # Compute OBV
    OBV = (last.diff() > 0).astype(int) * vol.diff()
    feature_table['OBV'] = OBV.cumsum()

    # Compute VWMA
    VWAP = (last * vol).cumsum() / vol.cumsum()
    feature_table['VWMA'] = VWAP.rolling(window=rolling).mean()

    # Compute FI
    feature_table['FI'] = last.diff(rolling) * vol.diff(rolling)

    return feature_table





