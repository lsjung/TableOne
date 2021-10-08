#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from datetime import date, datetime
import six
import yaml

with open("config.yml") as config:
    cfg = yaml.load(config, Loader=yaml.SafeLoader)
    
def read_data(datapath, subsetlist):
    """
    Read in data according to given path
    """
    inter_data1 = pd.read_csv(datapath)
    inter_data2 = inter_data1.drop_duplicates(subset=["USUBJID"])
    global tab1_data
    if subsetlist == None:
        tab1_data = inter_data2
    elif len(subsetlist)!=0:
        tab1_data = inter_data2.loc[inter_data2['SUBJID'].isin(subsetlist)]

def cont_med_t1(data, var, cutpoint = False, trt=False):
    """
    Median and range of the non-normal continuous variable
    
    Frequency and percentages based on 
    """
    med_row1 = pd.Series(data.groupby(['TRTA'])[var].median().apply(np.array).reset_index().iloc[:,1], name='med_row1')
    min_row1 = pd.Series(data.groupby(['TRTA'])[var].min().apply(np.array).reset_index().iloc[:,1], name='min_row1')
    max_row1 = pd.Series(data.groupby(['TRTA'])[var].max().apply(np.array).reset_index().iloc[:,1], name='max_row1')
    stats_row1 = pd.concat([header, med_row1, min_row1, max_row1], axis=1)
    
    # concatenate the median and range into one value
    for i in range(len(stats_row1)):
        stats_row1.loc[i,'median_range'] = str(round(med_row1[i],1)) + leftparen + str(round(min_row1[i],1)) + comma + str(round(max_row1[i],1)) + rightparen
    
    # strip dataframe to just the row with full information (median and range)
    stats_row1a = pd.DataFrame(stats_row1.iloc[:,np.r_[0,-1]]).T
    stats_row1a.columns = stats_row1a.iloc[0]
    stats_row1b = stats_row1a.drop([0], axis=0)
    stats_row1c = stats_row1b.reset_index()
    
    # cutpoint frequency and percentages 
    cutpoint = cutpoint
    efficacycut_row1 = data.copy()
    
    # create cutpoint intervals
    bins_row1 = [(efficacycut_row1[var].min()-1), cutpoint, (efficacycut_row1[var].max()+1)]
    labels_lorow1 = "< {}".format(cutpoint)
    labels_hirow1 = ">= {}".format(cutpoint)
    labels_row1 = [labels_lorow1, labels_hirow1]
    
    # create the binary variable based on cutpoint
    efficacycut_row1['row1'] = pd.cut(efficacycut_row1[var], bins=bins_row1, labels=labels_row1)
    efficacycut_row1.reset_index()

    # create a column for counts
    efficacycut_row1a = pd.DataFrame(efficacycut_row1.groupby(['TRTA','row1']).size())
    efficacycut_row1b = efficacycut_row1a.reset_index().rename(columns={0:'counts'}, inplace=False)

    # create a column for percentages
    efficacycut_row1c = round(efficacycut_row1a.groupby(level=0).apply(lambda x: 100*x/x.sum()), 1)
    efficacycut_row1d = efficacycut_row1c.reset_index().rename(columns={0:'pct'}, inplace=False)

    efficacycut_row1e = efficacycut_row1b.set_index(['TRTA','row1']).join(efficacycut_row1d.set_index(['TRTA','row1']))
    efficacycut_row1f = efficacycut_row1e.reset_index()

    for i in range(len(efficacycut_row1f)):
        efficacycut_row1f.loc[i, 'value'] = str(efficacycut_row1f.loc[i, 'counts']) + leftparen + str(efficacycut_row1f.loc[i, 'pct']) + rightparen

    efficacycut_row1g = efficacycut_row1f.pivot(index='TRTA',columns='row1')[['value']].T.reset_index()
    efficacycut_row1g.columns.name = None
    efficacycut_row1g
    efficacycut_row1h = efficacycut_row1g.drop(['level_0'], axis=1).rename(columns={'row1':'index'})
    efficacycut_row1h
    
    global var1
    var1 = stats_row1c.append(efficacycut_row1h)
    var1.index = [var, ' ', ' ']
    global table1
    table1 = tabulate(var1, header)


# output goal: a table of one row of median and range, a table of two rows by cutpoint and the frequency and % , and merge together 
# in the final nested function, we will make that 1 in table1 in to an iterable variable


def cont_mean_t1(data, var):
    """
    
    """
    mean_row1 = pd.Series(data.groupby(['TRTA'])[var].mean().apply(np.array).reset_index().iloc[:,1], name="mean_row1")
    std_row1 = pd.Series(data.groupby(['TRTA'])[var].std().apply(np.array).reset_index().iloc[:,1], name="std_row1")
    stats_row1 = pd.concat([header, mean_row1, std_row1], axis=1)
    
    # concatenate the mean and std into one value
    for i in range(len(stats_row1)):
        stats_row1.loc[i,'mean_std'] = str(round(mean_row1[i],1)) + leftparen + str(round(std_row1[i],2)) + rightparen
    
    # strip dataframe to just the row with full information (mean and std)
    stats_row1a = pd.DataFrame(stats_row1.iloc[:,np.r_[0,-1]]).T
    stats_row1a.columns = stats_row1a.iloc[0]
    stats_row1b = stats_row1a.drop([0], axis=0)
    stats_row1c = stats_row1b.reset_index()
    
    global var1
    var1 = stats_row1c
    #var1.index = np.array([var])
    var1.index = [var]
    global table1
    table1 = tabulate(var1, header)

# output goal: a table of one row of mean and standard deviation
# in the final nested function, we will make that 1 in table1 in to an iterable variable


def cat_t1(data, var):
    """
    
    """
    # create a column for counts
    efficacycut_row1a = pd.DataFrame(data.groupby(['TRTA',var]).size())
    efficacycut_row1b = efficacycut_row1a.reset_index().rename(columns={0:'counts'}, inplace=False)

    # create a column for percentages
    efficacycut_row1c = round(efficacycut_row1a.groupby(level=0).apply(lambda x: 100*x/x.sum()), 1)
    efficacycut_row1d = efficacycut_row1c.reset_index().rename(columns={0:'pct'}, inplace=False)

    efficacycut_row1e = efficacycut_row1b.set_index(['TRTA',var]).join(efficacycut_row1d.set_index(['TRTA',var]))
    efficacycut_row1f = efficacycut_row1e.reset_index()

    for i in range(len(efficacycut_row1f)):
        efficacycut_row1f.loc[i, 'value'] = str(efficacycut_row1f.loc[i, 'counts']) + leftparen + str(efficacycut_row1f.loc[i, 'pct']) + rightparen

    efficacycut_row1g = efficacycut_row1f.pivot(index='TRTA',columns=var)[['value']].T.reset_index()
    efficacycut_row1g.columns.name = None
    efficacycut_row1g
    efficacycut_row1h = efficacycut_row1g.drop(['level_0'], axis=1).rename(columns={var:'index'})
    efficacycut_row1h

    global var1
    var1 = efficacycut_row1h
    var1.index = [var] + ([' '] * (len(efficacycut_row1h)-1))
    global table1
    table1 = tabulate(var1, header)

#output goal: a table of one row of 
# try and make the first level of index named as the var, and each ow of the table as indented labels of the values of the unique categories

#output goal: a table of one row of 
# try and make the first level of index named as the var, and each ow of the table as indented labels of the values of the unique categories

def render_table_one(data, col_width=5.0, row_height=0.625, font_size=14, header_color="#830051", row_colors=["#E6CCDC", "w"], edge_color="w", bbox=[0,0,1,1], header_columns=0, ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax


def table_one(datapath, subsetlist, conmean, conmed, cat, cutpoint=65):
    """
    Create Table 1 based on the dataset and subset of subjects given. 
    
    Default setting is to produce descriptive summaries for AGE and SEX as those 2 are standard demographics variables expected in all datasets. 
        Cutpoint of 65 has been deemed based on MYSTIC and CASPIAN Table 1 cutpoints.
    """
    # headings including a blank column for the attribute labels
    elements = dir()
    del elements
    
    read_data(datapath, subsetlist)
    
    global dat1
    dat1 = tab1_data.groupby('TRTA').size().reset_index(name='counts')
    for i in range(len(dat1)):
        dat1.loc[i,'headers'] = dat1.loc[i, 'TRTA'] + "\n(n = " + str(dat1.loc[i, 'counts']) + ")"
    global header        
    header = pd.Series(np.unique(tab1_data['TRTA']))
    
    global leftparen
    leftparen = " ("
    global comma
    comma = ", "
    global rightparen
    rightparen = ")"

    global fintable
    fintable = pd.DataFrame()
    for i in range(len(conmean)):
        cont_mean_t1(tab1_data, conmean[i])
        fintable = fintable.append(var1)
    
    for i in range(len(conmed)):
        cont_med_t1(tab1_data, conmed[i], cutpoint[i])
        fintable = fintable.append(var1)
    
    for i in range(len(cat)):
        cat_t1(tab1_data, cat[i])
        fintable = fintable.append(var1)

    fintable = fintable.fillna("0")
    
    fintable = fintable.rename(columns={'index':''})
    dat1 = tab1_data.groupby('TRTA').size().reset_index(name='counts')
    for i in range(len(dat1)):
        dat1.loc[i,'headers'] = dat1.loc[i, 'TRTA'] + "\n(n = " + str(dat1.loc[i, 'counts']) + ")"   
    global header_n
    header_n = pd.Series(np.unique(dat1['headers']))
    # print(tabulate(fintable, header_n, tablefmt="pretty"))
    
    dat1 = fintable.reset_index()
    for i in range(len(dat1)):
        dat1.loc[i,'new_index'] = str(dat1.iloc[i,0]) + "   " +  str(dat1.iloc[i,1])

    global dat2
    dat2 = dat1.drop(['','index'],axis=1).set_index('new_index').reset_index()
    dat2.columns = [''] +list(header_n)

    render_table_one(dat2, header_columns=0, col_width=4.0)

    export_path = "Table 1. " + date.today().strftime("%Y%b%d") + " " + datetime.now().strftime("%H%M%S") + ".png"
    plt.savefig(export_path)

    

# output goal: loop through the list of variables in the arguments and then append all of the individual tables together

table_one(cfg['data'], subsetlist=cfg['subjects'], conmean=cfg['conmean'], conmed=cfg['conmed'], cat=cfg['cat'], cutpoint=cfg['cutpoint'])


