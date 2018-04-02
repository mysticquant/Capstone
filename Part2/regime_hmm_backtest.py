# regime_hmm_backtest.py


from __future__ import print_function

import datetime
import pickle
import warnings

from hmmlearn.hmm import GaussianHMM
from matplotlib import cm, pyplot as plt

from matplotlib.dates import YearLocator, MonthLocator

import numpy as np
import pandas as pd
import scipy as sp

from pandas_datareader import data as pdr
import pandas_datareader.data as web
import fix_yahoo_finance as yf
yf.pdr_override()
from scipy.spatial.distance import mahalanobis

from qstrader import settings
from qstrader.compat import queue
from qstrader.portfolio_handler import PortfolioHandler
from qstrader.position_sizer.naive import NaivePositionSizer
from qstrader.price_handler.yahoo_daily_csv_bar import \
YahooDailyCsvBarPriceHandler
from qstrader.price_parser import PriceParser
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading_session import TradingSession
from regime_hmm_strategy import MovingAverageCrossStrategy
#The one which doesnt use HMM information
#from regime_hmm_risk_manager import RegimeHMMRiskManager
#The one that uses HMM information in the proper form meant for investing
#Which means invest durng normal time, abstain during volatile times
from regime_hmm_risk_manager_with_hmm import RegimeHMMRiskManager
#This is the reverse of the above strategy
#from regime_hmm_risk_manager_with_hmm_reverse import RegimeHMMRiskManager


def mahalanobis_R(X,mean,IC):
    """
    This is the function that calculates the Mahalanobis distance
    Mahalanobis distance is the sort of distance from the centroid 
    of a cluster of points in finite space.
    """
    m =[]
    Mdist ={}
    for i in range(X.shape[0]):
        m.append(mahalanobis(X.ix[i,:],mean,IC)**2)
    Mdist["Mdist"] = m
    m = pd.DataFrame(Mdist,index=X.index)
    return m

def rolling_window_mahalanobis(X,window_length=0):
    """
    This calculates the mahalanobis window for different
    windows in history
    For the inflation and growth, no window period is used
    For Equity 2500 is used whereas for currency 750 is used
    """
    X.dropna(inplace=True)
    final = pd.DataFrame()
    for j in range(X.shape[1]):
        X1 = X.ix[:,j]
        
        if window_length != 0:
        
            splits = int(len(X)/window_length)
            splitter = X1.ix[:splits*window_length]
            a = np.split(splitter,splits)
            #a1 = X.ix[splits*window_length:]
        else:
            a = np.split(X,1)
        M_dist = []
        for i in a:
            if not isinstance(i, pd.DataFrame):
                i = pd.DataFrame(i)
            mean_i = np.mean(i)
            Cov_i = i.cov().values
            Cov_i= sp.linalg.inv(Cov_i)
            M_dist.append(mahalanobis_R(i,mean_i,Cov_i))
        M_dist = pd.concat(M_dist)
        final = pd.concat([final,M_dist],axis=1)
    final = pd.DataFrame(final.mean(axis=1))
    if window_length != 0:
        final = final.resample('M').mean()
        
    return final

def build_HMM_model2():
    
    sdt = datetime.datetime(1971, 1, 1)
    edt = datetime.datetime(2004, 12, 29)
    DEXUSAL = web.DataReader("DEXUSAL", "fred", sdt, edt)
    DEXUSUK = web.DataReader("DEXUSUK", "fred", sdt, edt)
    DEXCAUS = web.DataReader("DEXCAUS", "fred", sdt, edt)
    DEXNOUS = web.DataReader("DEXNOUS", "fred", sdt, edt)
    DEXJPUS = web.DataReader("DEXJPUS", "fred", sdt, edt)
    DEXUSNZ = web.DataReader("DEXUSNZ", "fred", sdt, edt)
    DEXSDUS = web.DataReader("DEXSDUS", "fred", sdt, edt)
    DEXSZUS = web.DataReader("DEXSZUS", "fred", sdt, edt)
    
    dfs = [DEXUSAL,DEXUSUK,DEXCAUS,DEXNOUS,DEXJPUS,DEXUSNZ,DEXSDUS,DEXSZUS]
    currency_df = pd.concat(dfs,axis=1)
    currency_df.columns = ['AUS','GBP','CAD','NOR','JAP','NZD','SWD','CHF']
    currency_df = currency_df.apply(pd.to_numeric, errors='coerce', axis=1)
    #For AUS,UK,NZ invert
    currency_df.AUS  = 1 / currency_df.AUS
    currency_df.GBP  = 1 / currency_df.GBP
    currency_df.NZD  = 1 / currency_df.NZD
    currency_df = np.log(currency_df) - np.log(currency_df.shift(1))
    
    sdt = datetime.datetime(1971, 1, 1)
    edt = datetime.datetime(2004, 12, 29)
    sym = "SPY"
    get_price = lambda sym, start, end: pdr.get_data_yahoo(sym, start, end)
    GSPC = pd.DataFrame(get_price(sym,sdt,edt))
    GSPC.to_csv("SPY_model2.csv")
    GSPC  = np.log(GSPC) - np.log(GSPC.shift(1))
    GSPC  = pd.DataFrame(GSPC["Adj Close"])
    GSPC.dropna(inplace=True)
    Turbulence_SPX= rolling_window_mahalanobis(GSPC,2500)
    Turbulence_currency = rolling_window_mahalanobis(currency_df,750)
    hmm_model_SPX = GaussianHMM(
        n_components=2, covariance_type="full", n_iter=1000
    ).fit(Turbulence_SPX)
    hmm_model_currency = GaussianHMM(
        n_components=2, covariance_type="full", n_iter=1000
    ).fit(Turbulence_currency)
    pickle_path_spy = "hmm_model_spy2.pkl"
    pickle_path_fx = "hmm_model_fx.pkl"
    print("Pickling HMM model 2 for SPY...")
    pickle.dump(hmm_model_SPX, open(pickle_path_spy, "wb"))
    print("...HMM model pickled.")
    print("Pickling HMM model for currency...")
    pickle.dump(hmm_model_currency, open(pickle_path_fx, "wb"))
    print("...HMM model pickled.")
    
    
def build_HMM_model():
     # Create the SPY dataframe from the Yahoo Finance CSV
    # and correctly format the returns for use in the HMM
    #This builds the model using direct returns
    file_name = "SPY_model.csv"
    pickle_path = "hmm_model_spy.pkl"
    try:
        spy = pd.read_csv(file_name)
        spy["Returns"] = spy["Adj Close"].pct_change()
        spy.dropna(inplace=True)
        rets = np.column_stack([spy["Returns"]])
    except IOError as error:
        get_price = lambda sym, start, end: pdr.get_data_yahoo(sym, start, end)
        
        #start_date = datetime.datetime(1993, 1, 29)
        #end_date = datetime.datetime(2010, 12, 29)
        start_date = datetime.datetime(2005, 1, 1)
        end_date = datetime.datetime(2010, 12, 29)
        sym = "SPY"
        spy = pd.DataFrame(get_price(sym,start_date,end_date))
        spy.to_csv("SPY_model.csv")
        spy["Returns"] = spy["Adj Close"].pct_change()
        
        spy.dropna(inplace=True)
        rets = np.column_stack([spy["Returns"]])
        

    # Create the Gaussian Hidden markov Model and fit it
    # to the SPY returns data, outputting a score
    hmm_model = GaussianHMM(
        n_components=2, covariance_type="full", n_iter=1000
    ).fit(rets)
    print("Model Score:", hmm_model.score(rets))

    # Plot the in sample hidden states closing values
    plot_in_sample_hidden_states(hmm_model, spy, rets)

    print("Pickling HMM model...")
    pickle.dump(hmm_model, open(pickle_path, "wb"))
    print("...HMM model pickled.")
    
def plot_in_sample_hidden_states(hmm_model,df, rets):
    """
    Plot the adjusted closing prices masked by
    the in-sample hidden states as a mechanism
    to understand the market regimes.
    """
    # Predict the hidden states array
    hidden_states = hmm_model.predict(rets)
    # Create the correctly formatted plot
    fig, axs = plt.subplots(
        hmm_model.n_components,
        sharex=True, sharey=True
    )
    colours = cm.rainbow(
        np.linspace(0, 1, hmm_model.n_components)
    )
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        ax.plot_date(
            df.index[mask],
            df["Adj Close"][mask],
            ".", linestyle='none',
            c=colour
        )
        ax.set_title("Hidden State #%s" % i)
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.grid(True)
    plt.show()

def run(config, testing, tickers, filename):
    #build_HMM_model()
    build_HMM_model2()
    title = ["Trend Following Regime Detection without HMM"]
    #pickle_path = "hmm_model_spy.pkl"
    pickle_path = "hmm_model_spy2.pkl"
    #pickle_path = "hmm_model_fx.pkl"
    events_queue = queue.Queue()
    csv_dir = config.CSV_DATA_DIR
    initial_equity = 100000.00
    start_date = datetime.datetime(2011, 1, 1)
    end_date = datetime.datetime(2018, 3, 31)
    # Use the Moving Average Crossover trading strategy
    base_quantity = 10000
    strategy = MovingAverageCrossStrategy(
            tickers, events_queue, base_quantity,
            short_window=10, long_window=30
            )
    # Use Yahoo Daily Price Handler
    price_handler = YahooDailyCsvBarPriceHandler(
            csv_dir, events_queue, tickers,
            start_date=start_date,
            end_date=end_date,
            calc_adj_returns=True
            )
    # Use the Naive Position Sizer
    # where suggested quantities are followed
    position_sizer = NaivePositionSizer()
    # Use an example Risk Manager
    #risk_manager = ExampleRiskManager()
    # Use regime detection HMM risk manager
    hmm_model = pickle.load(open(pickle_path, "rb"))
    risk_manager = RegimeHMMRiskManager(hmm_model)
    # Use the default Portfolio Handler
    portfolio_handler = PortfolioHandler(
             PriceParser.parse(initial_equity),
             events_queue, price_handler,
             position_sizer, risk_manager
             )
    # Use the Tearsheet Statistics class
    statistics = TearsheetStatistics(
            config, portfolio_handler,
            #title,benchmark = "EEM"
            title, benchmark="VWO"
            )
    
	# Set up the backtest
    backtest = TradingSession(
            config, strategy, tickers,
            initial_equity, start_date, end_date,
            events_queue, title=title,
            price_handler=price_handler,
            position_sizer=position_sizer,
            risk_manager=risk_manager,
            statistics=statistics,
            portfolio_handler=portfolio_handler
            )
    results = backtest.start_trading(testing=testing)
    return results

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # Configuration data
    testing = True
    config = settings.from_file(
            settings.DEFAULT_CONFIG_FILENAME, testing
            )
    #tickers = ["EEM","SPY"]
    tickers = ["VWO","SPY"]
    filename = None
    run(config, testing, tickers, filename)