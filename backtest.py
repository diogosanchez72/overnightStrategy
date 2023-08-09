##### Pacotes

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
import black_scholes as bs
from IPython.display import display, HTML,clear_output


### Recebe dataframe com [Macrostrategy , Substrategy, Ativo , Peso, Date, TradeType(Overnight)] outro dataframe com [Ativo, Date, Close]
# retorna dataframe com [Macrostrategy , Substrategy, Ativo , Peso, Date, TradeType(Overnight), Close, Return, ReturnOvernight]

