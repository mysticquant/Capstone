# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 12:14:33 2018

@author: sundar
"""

import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import numpy as np
from scipy.spatial.distance import mahalanobis
import scipy as sp
from hmmlearn.hmm import GaussianHMM
import warnings
import matplotlib.pyplot as plt
import pickle
import re


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



def Edhec_Returns(time_series):
    """
    This function plots the return time series of the 
    EDHEC hedge funds
    """
    
    a = time_series.copy()
    a.reset_index(inplace=True)
    a.columns = a.columns.str.replace('\s+', '_')
   
    
    for i in a.columns[1:]:

         fig,ax = plt.subplots(figsize=(10,8))
         ax.plot(a["date"],a[i])
         file_name = "EDHEC_"+i+".pickle"
         file_name = file_name.replace("/", "_")
         ax.set_title("EDHEC return for "+ i + " Strategy")
         with open(file_name, "wb") as f:
             pickle.dump(ax,f)
    

def fit_hmm(turb_series):
    """
    This module fits the HMM model 
    And also outputs some of the model results such 
    as the persistence probability and the transition probability
    A two state Gaussian model is used here
    """
    a = turb_series.copy()
    hmm_model = GaussianHMM(
        n_components=2, covariance_type="full", n_iter=1000
    ).fit(a)
    hidden_states = hmm_model.predict(a)
    initial_state = hidden_states[0]
    persistence_normal = hmm_model.transmat_[0][0]
    transition_normal = hmm_model.transmat_[0][1]
    mean_normal = hmm_model.means_[0][0]
    Std_Dev_normal = np.sqrt(hmm_model.covars_[0])[0][0]
    persistence_event = hmm_model.transmat_[1][1]
    transition_event = hmm_model.transmat_[1][0]
    mean_event = hmm_model.means_[1][0]
    Std_Dev_event = np.sqrt(hmm_model.covars_[1])[0][0]
    
    hmm_model_results = [initial_state,persistence_normal,transition_normal,\
                         mean_normal,Std_Dev_normal,persistence_event,\
                         transition_event,mean_event, Std_Dev_event]
    hidden_states = pd.DataFrame(hidden_states,\
                                 columns=["NormalorEventClass"],index=a.index)
    posterior_prob = hmm_model.predict_proba(a)
    posterior_prob = pd.DataFrame(posterior_prob,columns=["Event.Prob",\
    "Normal.Prob"],index=a.index)
    return pd.concat([a,posterior_prob,hidden_states],axis=1),hmm_model_results
    
def hmm_turbulence_plot(hmm_fit_dfs,name):
    """
    This makes a plot of the turbulence and the the different 
    Markovian states
    """
    plt.ioff()
    a = hmm_fit_dfs.copy()
    a1 = a.ix[:,0]
    a2 = a["NormalorEventClass"]
    
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,8))
    ax1.plot(a1,color="red")
    ax1.set_title(name)
    ax2.plot(a2,color="green")
    ax2.set_title("Event states")
    file_name = "Turbulence_plot_"+name+".pickle"
    with open(file_name, "wb") as f:
        pickle.dump((fig, (ax1,ax2)),f)
    
def estimate_plots_tables(table_normal,table_event):
    """
    This function creates tables and plots of the different parameters 
    estimated after the hmm model is fitted for the different regimes
    under different markovian states
    """
    a = table_normal.copy()
    b = table_event.copy()
    
    a = a.iloc[:,1:]
    b = b.iloc[:,1:]
    a["Regime"] = ["Equity","Currency","Inflation","Growth"]
    
    combined = pd.merge(a,b,right_index=True, left_index=True, \
                        suffixes=('_Normal', '_Event'))  
    persistence = combined[["Regime","Persistence_Normal","Persistence_Event"]]
    transition = combined[["Regime","Transition_Normal","Transition_Event"]]
    mean = combined[["Regime","Mean_Normal","Mean_Event"]]
    std_dev = combined[["Regime","Std_Dev_Normal","Std_Dev_Event"]]

    persistence.columns = ["Regime","Normal","Event"]
    mean.columns = ["Regime","Normal","Event"]
    transition.columns = ["Regime","Normal","Event"]
    std_dev.columns = ["Regime","Normal","Event"]
    
    ind = np.arange(len(mean)) 
    width = 0.4 
    fig, ax = plt.subplots()
    ax.barh(ind, mean.Normal, width, color='blue', label='Normal')
    ax.barh(ind + width, mean.Event, width, color='red', label='Event')

    ax.set(yticks=ind + width, yticklabels=mean.Regime, ylim=[2*width - 1,\
                                                               len(mean)])
    
    ax.set_title("Mean during different states")                                                                                                                   
    ax.legend()
   
    file_name = "Mean_Across_Regimes.pickle"
    with open(file_name, "wb") as f:
        pickle.dump(ax,f)
        
    file_name = "Mean_Across_Regimes_table.pickle"
    with open(file_name, "wb") as f:
        pickle.dump(mean,f)
    
    fig, ax = plt.subplots()
    ax.barh(ind, std_dev.Normal, width, color='blue', label='Normal')
    ax.barh(ind + width, std_dev.Event, width, color='red', label='Event')

    ax.set(yticks=ind + width, yticklabels=std_dev.Regime, ylim=[2*width - 1,\
                                                              len(std_dev)])
    ax.legend()
    ax.set_title("Std deviation during different states")
    file_name = "Standard_Deviation_Across_Regimes.pickle"
    with open(file_name, "wb") as f:
        pickle.dump(ax,f)
        
    file_name = "Standard_Deviation_Across_Regimes_table.pickle"
    with open(file_name, "wb") as f:
        pickle.dump(std_dev,f)
    
    fig, ax = plt.subplots()
    ax.barh(ind, persistence.Normal, width, color='blue', label='Normal')
    ax.barh(ind + width, persistence.Event, width, color='red', label='Event')

    ax.set(yticks=ind + width, yticklabels=persistence.Regime, ylim=[2*width - 1,\
                                                              len(persistence)])
    ax.legend()
    ax.set_title("Persistence probability during different states")
    file_name = "Persistence_probability_Across_Regimes.pickle"
    with open(file_name, "wb") as f:
        pickle.dump(ax,f)
        
    file_name = "Persistence_probability_Across_Regimes_table.pickle"
    with open(file_name, "wb") as f:
        pickle.dump(persistence,f)
    
    fig, ax = plt.subplots()
    ax.barh(ind, transition.Normal, width, color='blue', label='Normal')
    ax.barh(ind + width, transition.Event, width, color='red', label='Event')

    ax.set(yticks=ind + width, yticklabels=transition.Regime, ylim=[2*width - 1,\
                                                              len(transition)])
    ax.legend()
    ax.set_title("Transition probability during different states")
    file_name = "Transition_probability_Across_Regimes.pickle"
    with open(file_name, "wb") as f:
        pickle.dump(ax,f)
        
    file_name = "Transition_probability_Across_Regimes_table.pickle"
    with open(file_name, "wb") as f:
        pickle.dump(transition,f)
    
    return [mean,std_dev,persistence,transition]

def create_aligned_dataframes(equity_df,regime_df):
    EDF = equity_df.copy()
    Regime =  regime_df.copy()
    combined =  pd.merge(Regime,EDF,right_index=True, left_index=True)   
    return combined
    
def insample_performance(Aligned_df,title):
    """What this does it take the insample aligned returns 
    of different strategies under different regimes and  states 
    and calculates the overall performance for each regime
    """
    ADF = Aligned_df.copy()
    ADF_normal = ADF.copy()
    ADF_event = ADF.copy()
    ADF_normal.update(ADF_normal.iloc[:, 4:].\
                      mul(ADF_normal.NormalorEventClass, 0))
    ADF_event.iloc[:,3] = abs(1 - ADF_event.iloc[:,3]) 
    ADF_event.update(ADF_event.iloc[:, 4:].\
                      mul(ADF_event.NormalorEventClass, 0))
    ADF_normal_mean = ADF_normal.iloc[:,4:].mean(axis =0)
    ADF_event_mean = ADF_event.iloc[:,4:].mean(axis =0)
    ADF_sd = ADF.iloc[:,4:].std(axis=0)
    ADF_perf = (ADF_event_mean - ADF_normal_mean)/ADF_sd
    ADF_perf = pd.DataFrame(ADF_perf)
    ADF_perf.reset_index(inplace=True)
    ADF_perf.columns = ["Fund","Perf"]
    ind = np.arange(len(ADF_perf)) 
    width = 0.4 
    fig, ax = plt.subplots(figsize=(10,8))
    ax.barh(ind, ADF_perf.Perf, width, color='blue')
    

    ax.set(yticks=ind + width, yticklabels=ADF_perf.Fund, ylim=[2*width - 1,\
                                                              len(ADF_perf)])
    ax.legend()
    ax.set_title("Scaled Difference in Mean "+title)
    file_name = "Scaled_Difference_in_Mean_"+title+".pickle"
    with open(file_name, "wb") as f:
        pickle.dump(ax,f)
    return ADF_normal_mean, ADF_event_mean,ADF_perf
    
def drawdown(X):
    peak = X[0]
    dd = np.zeros(len(X))
    i = 0
    for x in X:
        if x > peak: 
            peak = x
        #dd = (peak - x) / peak # This is the original definition
        #Prefer to use this definition
        #This is more like a  drawdown used in accounting
        dd[i] = x - peak
        i = i + 1
        
    return dd
    
def comparison_plots(Aligned_df,title):
    """
    What this does is it plots the cumulative returns for each fund for each
    regime . It plots the cumulative returns and the drawdowns with and without
    regime markovian state information. Here the assumption is we invest during
    the normal state and we dont during the event state
    """
    ADF = Aligned_df.copy()
    ADF_original =ADF.copy()
    #This has the Markovian information
    ADF.update(ADF.iloc[:, 4:].\
                      mul(ADF.NormalorEventClass, 0))
    ADF = ADF.iloc[:,4:]
    ADF_original = ADF_original.iloc[:,4:]
    
    combined =  pd.merge(ADF_original,ADF,right_index=True, left_index=True,\
    suffixes=('_Original', '_Markovian'))  
    #create unique list of names
    UniqueNames = ADF.columns

    #create a data frame dictionary to store your data frames
    DataFrameDict = {elem : pd.DataFrame() for elem in UniqueNames}

    for key in DataFrameDict.keys():
        original = key+"_Original"
        duplicate = key + "_Markovian"
        DataFrameDict[key] = combined[[original,duplicate]]
        DataFrameDict[key].columns = ["Original","Markovian"]
        DataFrameDict[key]["original_cumulative"] = \
        (1+DataFrameDict[key]["Original"]).cumprod()
        DataFrameDict[key]["markovian_cumulative"] = \
        (1+DataFrameDict[key]["Markovian"]).cumprod()
        #Drawdonw of returns
        DataFrameDict[key]["original_dd"] = \
        drawdown(np.array(DataFrameDict[key]["Original"]))
        DataFrameDict[key]["markovian_dd"] = \
        drawdown(np.array(DataFrameDict[key]["Markovian"]))
        
        
        key_string = key.replace("/", "_")
        key_string = re.sub(r"\s+", '_', key_string)

        #Pickle the plots
        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot(DataFrameDict[key][["original_cumulative","markovian_cumulative"]])
        ax.legend(["original","markovian"])
        ax.set_title("Cumulative returns plot for "+ key + " under "+ title)
        
        file_name = key_string+"_"+title+"_cumulative_returns.pickle"
        with open(file_name, "wb") as f:
            pickle.dump(ax, f)
        
        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot(DataFrameDict[key][["original_dd","markovian_dd"]])
        ax.legend(["original","markovian"])
        ax.set_title("Drawdowns plot for "+ key + " under "+ title)
        
        file_name = key_string+"_"+title+"_drawdowns.pickle"
        with open(file_name, "wb") as f:
            pickle.dump(ax, f)

def comparison_plots_across_regimes(first_df,*args):
    Main_df = first_df.copy()
    for df in args:
        Main_df = pd.merge(Main_df,df,right_index=True, left_index=True)
    
    Main_df = Main_df.transpose()
    UniqueNames = Main_df.columns
    DataFrameDict = {elem : pd.DataFrame(columns = ["Regime","Normal","Event"]) for elem in UniqueNames}
    for key in DataFrameDict.keys():
        df = Main_df[key].copy()
        DataFrameDict[key]["Regime"]= ["Equity","Currency","Inflation","Growth"]
        DataFrameDict[key].iloc[0,1] = df.loc["Equity_Normal"]
        DataFrameDict[key].iloc[0,2] = df.loc["Equity_Event"]
        DataFrameDict[key].iloc[1,1] = df.loc["Currency_Normal"]
        DataFrameDict[key].iloc[1,2] = df.loc["Currency_Event"]
        DataFrameDict[key].iloc[2,1] = df.loc["Inflation_Normal"]
        DataFrameDict[key].iloc[2,2] = df.loc["Inflation_Event"]
        DataFrameDict[key].iloc[3,1] = df.loc["Growth_Normal"]
        DataFrameDict[key].iloc[3,2] = df.loc["Growth_Event"]
        
        ind = np.arange(len(DataFrameDict[key])) 
        width = 0.4 
    
        fig, ax = plt.subplots(figsize=(10,8))
        ax.barh(ind, DataFrameDict[key].Normal, width, color='blue', label='Normal')
        ax.barh(ind + width, DataFrameDict[key].Event, width, color='red', label='Event')

        ax.set(yticks=ind + width, yticklabels=DataFrameDict[key].Regime, ylim=[2*width - 1,\
                                                              len(DataFrameDict[key])])
        ax.legend()
        key_string = key.replace("/", "_")
        key_string = re.sub(r"\s+", '_', key_string)
        ax.set_title("Comparison plots for strategy "+key_string +\
                     " Across different regimes")
        
        file_name = key_string+"_comparison_plots.pickle"
        with open(file_name, "wb") as f:
            pickle.dump(ax, f)
        
        file_name = key_string+"_comparison_plots_table.pickle"
        with open(file_name, "wb") as f:
            pickle.dump(DataFrameDict[key], f)
        
        
if __name__ == "__main__":
    # Hides deprecation warnings for sklearn
    warnings.filterwarnings("ignore")
    
    #Load the regime variables
    #S&P 500
    GSPC = pd.read_csv("GSPC.csv",index_col=[0], parse_dates=True)
    #Using log returns
    GSPC  = np.log(GSPC) - np.log(GSPC.shift(1))
    GSPC  = pd.DataFrame(GSPC["GSPC.Adjusted"])
    #Various currency 
    sdt = dt.datetime(1971, 1, 1)
    edt = dt.datetime(2018, 3, 1)
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
    
    
    #Inflation turbulence
    sdt = dt.datetime(1946, 1, 1)
    edt = dt.datetime(2018, 3, 1)
    CPIAUCNS = web.DataReader("CPIAUCNS", "fred", sdt, edt)
    CPIAUCNS = np.log(CPIAUCNS) - np.log(CPIAUCNS.shift(1))
    #In turbulence CPIAUCNS one outlier is so much skewing the HMM
    #Remmoving the outlier. It is 1940s so wont affect result
    
    
    CPIAUCNS = CPIAUCNS.drop(CPIAUCNS.index[[6]])
    #Economic turbulence
    GDPC1 =  web.DataReader("GDPC1", "fred", sdt, edt)
    GDPC1 = np.log(GDPC1) - np.log(GDPC1.shift(1))    
    #Calculate the Turbulence
    Turbulence_GDPC= rolling_window_mahalanobis(GDPC1)
    Turbulence_SPX= rolling_window_mahalanobis(GSPC,2500)
    
    
    
    Turbulence_CPIAUCNS = rolling_window_mahalanobis(CPIAUCNS)
    Turbulence_currency = rolling_window_mahalanobis(currency_df,750)
    
    
    #For Risk Premia I am using the EDHEC hedge fund indices
    Risk_premia = pd.read_csv("history.csv",index_col=[0], parse_dates=True)
    Edhec_Returns(Risk_premia)
    
    #Fit the HMM model
    HMM_GDPC,model_GDPC = fit_hmm(Turbulence_GDPC)
    #HMM_SPX,model_SPX = fit_hmm(Turbulence_SPX)
    #Similarly in SPX, the index corresponding to 1987-10-31(Black Friday) heavily skews
    #the HMM model. The HMM model is not robust enough to mark it as an outlier
    #It totally skews the states discovery
    #The HMM model is not robust enough to detect it
    # I have to use the model fitted in R by depmixS4 package 
    #Source code attached  
    #It could also be an entirely different state (state 3). But since we are 
    #dealing with only two. But still the equity data is available only from 1997
    
    HMM_SPX = pd.read_csv("HMM.csv",index_col=[0], parse_dates=True)
    #Similarly the values for SPX are taken from the R model
    model_SPX= [1,0.965,0.035,0.644,0.393,0.753,0.247,3.451,4.110]
    HMM_CPIAUCNS,model_CPIAUCNS = fit_hmm(Turbulence_CPIAUCNS)
    HMM_currency,model_currency = fit_hmm(Turbulence_currency)
    
    #HMM plots
    hmm_turbulence_plot(HMM_SPX,"Equity")
    hmm_turbulence_plot(HMM_currency,"Currency")
    hmm_turbulence_plot(HMM_CPIAUCNS,"Inflation")
    hmm_turbulence_plot(HMM_GDPC,"Growth")
    
    Turbulence_table = pd.DataFrame([model_SPX,model_currency,model_CPIAUCNS,\
                                     model_GDPC])
    Turbulence_table_normal = Turbulence_table.iloc[:,:5]
    Turbulence_table_event =  Turbulence_table.iloc[:,[0,5,6,7,8]]

    Turbulence_table_normal.columns=["Initial_State","Persistence","Transition",\
    "Mean", "Std_Dev"]
    
    Turbulence_table_event.columns=["Initial_State","Persistence","Transition",\
    "Mean", "Std_Dev"]
    
    #Create some plots and tables here
    estimate_plots_tables(Turbulence_table_normal,Turbulence_table_event)
    
    Aligned_Equity = create_aligned_dataframes(Risk_premia,HMM_SPX)
    Aligned_currency = create_aligned_dataframes(Risk_premia,HMM_currency)
    HMM_GDPC.index =HMM_GDPC.index.to_period('M').to_timestamp('M')
    HMM_CPIAUCNS.index =HMM_CPIAUCNS.index.to_period('M').to_timestamp('M')
    Aligned_Inflation = create_aligned_dataframes(Risk_premia,HMM_CPIAUCNS)
    Aligned_Growth = create_aligned_dataframes(Risk_premia, HMM_GDPC)
    
    AL_Equity_Normal,AL_Equity_Event,AL_Equity_Perf = \
    insample_performance(Aligned_Equity,"Equity") 
    
    AL_currency_Normal,AL_currency_Event,AL_currency_Perf = \
    insample_performance(Aligned_currency,"Currency") 
    
    AL_Inflation_Normal,AL_Inflation_Event,AL_Inflation_Perf = \
    insample_performance(Aligned_Inflation,"Inflation") 
    
    AL_Growth_Normal,AL_Growth_Event,AL_Growth_Perf = \
    insample_performance(Aligned_Growth,"Growth") 
    
    #Comparison plots
    comparison_plots(Aligned_Equity,"Equity")
    comparison_plots(Aligned_currency,"Currency")
    comparison_plots(Aligned_Inflation,"Inflation")
    comparison_plots(Aligned_Growth,"Growth")
    
    #Comparison plots across regimes
    Equity_Normal = pd.DataFrame(AL_Equity_Normal,columns=["Equity_Normal"])
    Equity_Event  = pd.DataFrame(AL_Equity_Event,columns=["Equity_Event"])
    Currency_Normal = pd.DataFrame(AL_currency_Normal,columns=["Currency_Normal"])
    Currency_Event = pd.DataFrame(AL_currency_Event,columns=["Currency_Event"])
    Inflation_Normal = pd.DataFrame(AL_Inflation_Normal,columns=["Inflation_Normal"])
    Inflation_Event = pd.DataFrame(AL_Inflation_Event,columns=["Inflation_Event"])
    Growth_Normal = pd.DataFrame(AL_Growth_Normal,columns=["Growth_Normal"])
    Growth_Event = pd.DataFrame(AL_Growth_Event,columns=["Growth_Event"])
    
    comparison_plots_across_regimes(Equity_Normal,Equity_Event,Currency_Normal,\
                                    Currency_Event,Inflation_Normal,Inflation_Event,\
                                    Growth_Normal,Growth_Event)
    
    
    
      
    
    
    
    
    
