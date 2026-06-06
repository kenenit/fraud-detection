import pandas as pd


def add_time_features(df):
    """Add hour_of_day and day_of_week from purchase_time."""
    df = df.copy()
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    return df


def add_time_since_signup(df):
    """Add time_since_signup in hours."""
    df = df.copy()
    df['time_since_signup'] = (
        df['purchase_time'] - df['signup_time']
    ).dt.total_seconds() / 3600
    return df


def add_transaction_velocity(df):
    """Add user_tx_count: number of transactions per user."""
    df = df.copy()
    tx_count = df.groupby('user_id')['purchase_time'].count().reset_index()
    tx_count.columns = ['user_id', 'user_tx_count']
    df = df.merge(tx_count, on='user_id', how='left')
    return df


def engineer_all_features(df):
    """Apply all feature engineering steps to fraud dataframe."""
    df = add_time_features(df)
    df = add_time_since_signup(df)
    df = add_transaction_velocity(df)
    return df


def drop_unused_columns(df):
    """Drop columns not needed for modeling."""
    cols_to_drop = [
        'user_id', 'device_id', 'ip_address', 'ip_int',
        'signup_time', 'purchase_time',
        'lower_bound_ip_address', 'upper_bound_ip_address'
    ]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])