from itertools import combinations
from typing import Optional, Union, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def create_interaction_features(df: pd.DataFrame, columns: Optional[Union[List[str], None]] = None):
    df = df.copy()
    if columns is None:
        columns = df.columns
    for col1, col2 in combinations(columns, 2):
        df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
        df[f"{col1}_+_{col2}"] = df[col1] - df[col2]
        df[f"{col1}_-_{col2}"] = df[col1] + df[col2]
        df[f"{col1}_/_{col2}"] = df[col1] / df[col2]

    return df


def create_polynomial(df, columns, degree=2, interaction_only=False, include_bias=False):
    df = df.copy()
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    if columns is None:
        columns = df.columns
    df_poly = pd.DataFrame(poly.fit_transform(df[columns]))
    df_poly.columns = ['poly_' + col if 'x' in col else col for col in poly.get_feature_names(columns)]
    return pd.concat([df, df_poly], axis=1)


def binning(df: pd.DataFrame, columns: List[str], n_bins: Union[int, str]='auto'):
    df = df.copy()
    df_binned = pd.DataFrame()
    for col in columns:
        #print(len(df))
        #values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        #print(len(values))
        values = df[col]
        bin_edges = np.histogram_bin_edges(values, bins=n_bins)
        labels = [i for i in range(len(bin_edges) - 1)]
        binned_values = pd.cut(df[col], bins=bin_edges, labels=labels).cat.codes.astype(int)
        df_binned = pd.concat([df_binned, binned_values], axis=1)
    df_binned.columns = [f'bin_{col}' for col in columns]
    return pd.concat([df, df_binned], axis=1)


