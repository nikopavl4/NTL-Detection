import numpy as np
from scipy.special import boxcox
from scipy.stats import yeojohnson


def log_transform(data):
    return np.log1p(data)


def sqrt_transform(data):
    return np.sqrt(data)


def box_cox_transform(data, lmbda=None):
    transformed_data, fitted_lambda = boxcox(data, lmbda=lmbda)
    return transformed_data, fitted_lambda


def yeo_johnson_transform(data, lmbda=None):
    transformed_data, fitted_lambda = yeojohnson(data, lmbda=lmbda)
    return transformed_data, fitted_lambda


def inverse_transform(data):
    return 1 / data


def identity_transform(data):
    return data
