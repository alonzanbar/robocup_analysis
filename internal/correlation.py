import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

def save_scatter(df,x,y,out_folder):
    pearson_corr =df[[x,y]].corr().iloc[0,1]
    spearman = df[[x,y]].corr(method='spearman').iloc[0,1]
    plt.scatter(data=df,x=x,y=y)
    x_title = " ".join(x.split('_')[1:])
    y_title = " ".join(y.split('_')[1:])
    plt.title("{0} vs {1} \n pearson: {2:.2f}, spearman: {3:.2f}".format(y_title,x_title,pearson_corr,spearman))
    plt.savefig(out_folder+"/"+"{}_{}.eps".format(y_title,x_title).replace(" ","_"),format='eps')
    plt.show()

def save_corrlation_matrix(title,df,method,x_pref,y_pref,out_folder,x_axis_title_size=0.15,figsize=(9,5),linewidths=.5):
    df_corr = df.corr(method=method)
    df_corr = df_corr[df_corr.index.str.contains(y_pref)]
    df_corr = df_corr[[a for a in df_corr if re.match(x_pref,a)]]
    df_corr = df_corr[~df_corr.isnull().all(axis=1)]
    
    df_corr = df_corr.T[~df_corr.T.isnull().all(axis=1)]
    df_corr_pretty = df_corr.copy()
    df_corr_pretty.columns = [a.replace(y_pref," ").replace("_"," ") for a in df_corr_pretty.columns]
    df_corr_pretty= df_corr_pretty.round(2).T
    df_corr_pretty.columns = [a.replace(x_pref," ").replace("_"," ") for a in df_corr_pretty.columns]
    f, ax = plt.subplots(figsize=figsize)
    plt.gcf().subplots_adjust(bottom=x_axis_title_size)
    sns_plot = sns.heatmap(df_corr_pretty,annot=True,linewidths=linewidths)
    sns_plot.set_title(
        '%s code analysis metrics to perfromance metrics corrlation matrix (%s)'%(title,method))
    sns_plot.figure.savefig(out_folder+"/"+"correlation_{}_{}_{}_{}.eps".format(title,y_pref,x_pref,method),format='eps')
    return df_corr