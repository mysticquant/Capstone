# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 19:20:03 2018

@author: sundar
"""

#This file basically creates a pdf file out of all those pickled
#files which were created using Regime_Detection.py
#This has to be run in a separate python sesssion
#Otherwise it will flag error
#The best way to run this code is open a command shell and invoke it

import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import six
import numpy as np
import pandas as pd

#Function for rendering dataframe as a fig in matplotlib
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

def truncate(x):
    if isinstance(x, float):
        return np.round(x, decimals=4)
    else:
        return x

if __name__ == "__main__":
  
    with PdfPages('Results.pdf') as pdf:
        #The first four plots are the turbulence plots of the regimes
        #and the Markovian states
        choices = ["Equity","Currency","Inflation","Growth"]
        for choice in choices:
            file_name = "Turbulence_plot_"+choice+".pickle"
            with   open( file_name, "r" ) as fid:
                ax = pickle.load(fid)
                plt.suptitle("Plot of Turbulence of "+ choice+ " and its States")
            pdf.savefig()
            
        #The next to go in the pdf would be the estimates of mean, sigma 
        #and probabilities of the different regimes
        choices = ["Mean","Standard_Deviation","Persistence_Probability","Transition_Probability"]
        for choice in choices:
            file_name = choice+"_Across_Regimes_table.pickle"
            with   open( file_name, "r" ) as fid:
                df = pickle.load(fid)
                #This is the table
                ax = render_mpl_table(df)
                plt.suptitle(choice+" Estimates in different regimes ")
                            
            pdf.savefig()
            #This is the bar chart
            file_name = choice+"_Across_Regimes.pickle"
            with   open( file_name, "r" ) as fid:
                ax = pickle.load(fid)
            pdf.savefig()   
             
        #Next comes the performance of different strategies 
        #First plot the returns of the strategies
        plt.close("all")
        Risk_premia = pd.read_csv("history.csv",index_col=[0], parse_dates=True)
        choices = Risk_premia.columns
        choices = choices.str.replace('\s+', '_')
        choices2 = ["Equity","Currency","Inflation","Growth"]
        for choice in choices:
            choice = choice.replace('/', '_')
            #First is the general returns of the strategy
            file_name = "EDHEC_"+choice+".pickle"
            with   open( file_name, "r" ) as fid:
                ax = pickle.load(fid)
                
            pdf.savefig()
            plt.close('all')
            #Second is the performance of the strategy under the different
            #regimes under different Markovian states
            #First will be the table
            file_name = choice+"_comparison_plots_table.pickle"
            with   open( file_name, "r" ) as fid:
                df = pickle.load(fid)
                df = df.apply(lambda x: truncate(x) ) 
                #This is the table
                ax = render_mpl_table(df)
                plt.suptitle(choice+" Estimates in different regimes ")
                            
            pdf.savefig()
            plt.close('all')
            #Next the figure
            file_name = choice+"_comparison_plots.pickle"
            with   open( file_name, "r" ) as fid:
                ax = pickle.load(fid)
                
            pdf.savefig()
            plt.close('all')
            
            for choice2 in choices2: 
            
                #Then plot the drawdown and cumulative return plots for 
                #the strategy under the different regimes
                file_name = choice+"_"+choice2+"_cumulative_returns.pickle"
                with   open( file_name, "r" ) as fid:
                    ax = pickle.load(fid)
                pdf.savefig()
                plt.close('all')   
                
                file_name = choice+"_"+choice2+"_drawdowns.pickle"
                with   open( file_name, "r" ) as fid:
                    ax = pickle.load(fid)
                pdf.savefig()
                plt.close('all') 
        #The final plots will be the the performance of all strategies under a 
        #regime
        choices = ["Equity","Currency","Inflation","Growth"]
        for choice in choices:
            file_name = "Scaled_Difference_in_Mean_"+choice+".pickle"
            with   open( file_name, "r" ) as fid:
                ax = pickle.load(fid)
            pdf.savefig()    
            plt.close('all') 
            