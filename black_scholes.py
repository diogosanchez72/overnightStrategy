import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import os
import pandas as pd
import requests
import seaborn as sns
import sys
import warnings
import yfinance as yf
from scipy import stats
import risk_functions as rf
from scipy.optimize import minimize

#Black Scholes option call precification
def bs_call(S, K, T, r, sigma):
    '''black_scholes_call(S, K, T, r, sigma)
    S: spot price
    K: strike price
    T: time to maturity
    r: interest rate
    sigma: volatility of underlying asset
    '''
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = (S * stats.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))

    if type(T) == np.ndarray:
        call = np.where(T < 1/250, S - K, call)

    elif type(T) == float:
        if T < 1/250:
            call = S - K
            call = np.where(call < 0, 0, call)
    
    return call


#Black Scholes option greek delta
def bs_call_delta(S,K,T,r,sigma):
    '''bs_call_delta(S,K,T,r,sigma)
    S: spot price
    K: strike price
    T: time to maturity
    r: interest rate
    sigma: volatility of underlying asset
    '''
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    delta = stats.norm.cdf(d1, 0.0, 1.0)
    return delta



#Black Scholes option greek gamma
def bs_call_gamma(S,K,T,r,sigma):
    '''bs_call_gamma(S,K,T,r,sigma)
    S: spot price
    K: strike price
    T: time to maturity
    r: interest rate
    sigma: volatility of underlying asset
    '''
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    gamma = stats.norm.pdf(d1, 0.0, 1.0) / (S * sigma * np.sqrt(T))
    return gamma

#Black Scholes greek vega
def bs_call_vega(S,K,T,r,sigma):
    '''bs_call_vega(S,K,T,r,sigma)
    S: spot price
    K: strike price
    T: time to maturity
    r: interest rate
    sigma: volatility of underlying asset
    '''
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    vega = S * stats.norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
    return vega

#Black Scholes greek theta
def bs_call_theta(S,K,T,r,sigma):
    '''bs_call_theta(S,K,T,r,sigma)
    S: spot price
    K: strike price
    T: time to maturity
    r: interest rate
    sigma: volatility of underlying asset
    '''
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    theta = -(S * stats.norm.pdf(d1, 0.0, 1.0) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0)
    return theta

#Black scholes greek rho
def bs_call_rho(S,K,T,r,sigma):
    '''bs_call_rho(S,K,T,r,sigma)
    S: spot price
    K: strike price
    T: time to maturity
    r: interest rate
    sigma: volatility of underlying asset
    '''
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0)
    return rho