import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(fraud_path, ip_path, cc_path):
    """Load all three raw datasets."""
    fraud_df = pd.read_csv(fraud_path)
    ip_df = pd.read_csv(ip_path)
    cc_df = pd.read_csv(cc_path)
    return fraud_df, ip_df, cc_df


def fix_dtypes(df):
    """Convert timestamp columns to datetime."""
    df = df.copy()
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df


def remove_duplicates(df):
    """Drop duplicate rows and reset index."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    print(f"Removed {before - after} duplicate rows.")
    return df


def ip_to_int(ip):
    """Convert dotted-decimal IP address to integer."""
    try:
        parts = str(ip).split('.')
        return sum(int(p) * (256 ** (3 - i)) for i, p in enumerate(parts))
    except Exception:
        return None


def merge_ip_country(fraud_df, ip_df):
    """
    Enrich fraud_df with country via range-based IP lookup.
    Uses pd.merge_asof for efficient O(n log n) range join.
    """
    fraud_df = fraud_df.copy()
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)
    fraud_df = fraud_df.dropna(subset=['ip_int'])
    fraud_df['ip_int'] = fraud_df['ip_int'].astype(int)

    ip_df = ip_df.copy()
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(float).astype(int)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(float).astype(int)
    ip_df_sorted = ip_df.sort_values('lower_bound_ip_address').reset_index(drop=True)

    fraud_sorted = fraud_df.sort_values('ip_int').reset_index(drop=True)

    merged = pd.merge_asof(
        fraud_sorted,
        ip_df_sorted[['lower_bound_ip_address', 'upper_bound_ip_address', 'country']],
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )

    merged['country'] = merged.apply(
        lambda r: r['country']
        if pd.notna(r['upper_bound_ip_address'])
        and r['ip_int'] <= r['upper_bound_ip_address']
        else 'Unknown',
        axis=1
    )
    return merged


def scale_features(df, num_cols):
    """Apply StandardScaler to specified numerical columns."""
    df = df.copy()
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, scaler


def encode_categoricals(df, cat_cols):
    """One-hot encode categorical columns."""
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)


def preprocess_creditcard(cc_df):
    """Scale Amount and Time in the credit card dataset."""
    cc_df = cc_df.copy()
    scaler = StandardScaler()
    cc_df['Amount_scaled'] = scaler.fit_transform(cc_df[['Amount']])
    cc_df['Time_scaled'] = scaler.fit_transform(cc_df[['Time']])
    cc_df = cc_df.drop(columns=['Amount', 'Time'])
    return cc_df