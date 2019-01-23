import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns
from math import sqrt
from scipy import stats as stats
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches

# Local imports
from util.histogram_bin_formulas import bin_it

def dataframe_plot(df, extra_test_dict = {}, quantiles=(0.05, 0.95), rare_category_factor=0.1,  rows_per_pixel=1, show_value_ranks = True):
    """
    Plots a dataframe with colours showinv various anomalous entries
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to plot
    quantiles : (float, float)
        Values between 0 and 1 giving upper and lower quantile levels for high and low values
    rare_category_factor : float
        Proportion of mean category count required for a category to be considered rare
    rows_per_pixel : float
        Number of rows that represent a pixel along the vertical axis. Use this to make the image shorter (but less representative)
    """
    na_colour = np.array([255,0,0])/255
    large_colour = np.array([255, 255, 255])/255
    small_colour = np.array([0, 0, 0])/255
    neg1_colour = np.array([255,200,0])/255
    rare_colour = np.array([127,0,255])/255
    emptystr_colour = np.array([255,128,0])/255

    img = np.ones((df.shape[0], df.shape[1], 3))*1/2
    img[df.isna()] = na_colour
    for i, c in enumerate(df.columns):
        if len(df[c].unique()) < np.ceil(df.shape[0]/10):
            counts = df[c].value_counts()
            img[df[c].isin(counts[counts < rare_category_factor*df.shape[0]/counts.shape[0]].index), i] = rare_colour
        if np.issubdtype(df[c], np.number):
            if show_value_ranks:
                ranks = stats.rankdata(df[c].values)
                img[:,i,:] = np.repeat((2/5 + (1/5)*(ranks-ranks.min())/(ranks.max()-ranks.min()))[:, np.newaxis], 3, axis=1)
            img[df[c] < df[c].quantile(quantiles[0]), i] = small_colour
            img[df[c] > df[c].quantile(quantiles[1]), i] = large_colour
            img[df[c] == -1, i] = neg1_colour
    img[df == ""] = emptystr_colour

    for key in extra_test_dict.keys():
        img[df == key,:] = extra_test_dict[key]

    fig = plt.gcf()
    fig.set_size_inches(0.3*df.shape[1], df.shape[0]/(fig.dpi*rows_per_pixel))
    plt.imshow(img, aspect='auto')
    plt.grid(axis='x', color='black', linewidth=0.5)
    ax = plt.gca()
    ax.set_xticks(-0.5+np.arange(0, df.shape[1], 1))
    plt.xticks(rotation=90, ha='left')
    ax.set_xticklabels(df.columns)

    patches = [
        mpatches.Patch(color=large_colour, label="Large value"),
        mpatches.Patch(color=small_colour, label="Small value"),
        mpatches.Patch(color=rare_colour, label="Rare value"),
        mpatches.Patch(color=na_colour, label="N/A"),
        mpatches.Patch(color=neg1_colour, label="-1"),
        mpatches.Patch(color=emptystr_colour, label="\"\""),
        ]
    for label, color in extra_test_dict.items():
        patches.append(mpatches.Patch(color=color, label=label))
    plt.legend(handles=patches, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0. )
    plt.tight_layout()


def dist_compare_plot(df_train, df_test, columns=None,
                      histogram=False, kde=True, rug=False, bins={},
                      xlim_quantiles=(0.025,0.975),
                      max_categories=40, subfigsize=(8,4),
                      dpi=150, hist_kws = {"alpha": 0.5},
                      kde_kws = {"shade": True}):
    if columns is None:
        columns = [c for c in df_train.columns if c in df_test.columns]
    else:
        columns = [c for c in columns if c in df_train.columns and c in df_test.columns]
    df_all = pd.concat([df_train[columns], df_test[columns]])
    if len(columns) < 1:
        raise ValueError("No numeric features to plot.")
    if type(bins) is int:
        bins = {c:bins for c in columns if np.issubdtype(df_all[c], np.number)}
    elif type(bins) is str:
        bins = {c:bin_it(df_all[c], bins) for c in columns if np.issubdtype(df_all[c], np.number)}
    elif type(bins) is dict:
        for c in [c for c in bins.keys() if type(bins[c]) is str]:
            bins[c] = bin_it(df_all[c].dropna().values, bins[c])
        for c in [c for c in bins.keys() if type(bins[c]) is int]:
            bins[c] = np.linspace(df_all[c].min(), df_all[c].max(), bins[c])
    subplot_cols = int(np.ceil(sqrt(len(columns))))
    subplot_rows = int(np.ceil(len(columns)/subplot_cols))
    fig = plt.figure(dpi=dpi, figsize=(subfigsize[0]*subplot_cols, subfigsize[1]*subplot_rows))
    for i, c in enumerate(columns):
        plt.subplot(subplot_rows, subplot_cols, i+1)
        plt.title(str(c))
        if np.issubdtype(df_train[c], np.number) and np.issubdtype(df_test[c], np.number):
            if c not in bins.keys():
                bins[c] = bin_it(df_all[c].dropna().values)
            sns.distplot(
                df_train[c].dropna(), hist=histogram, kde=kde,
                bins=bins[c], hist_kws=hist_kws,
                kde_kws=kde_kws, rug=rug
            )
            sns.distplot(
                df_test[c].dropna(), hist=histogram, kde=kde,
                bins=bins[c], hist_kws=hist_kws,
                kde_kws=kde_kws, rug=rug
            )
            ll = min(df_train[c].quantile(xlim_quantiles[0]), df_test[c].quantile(xlim_quantiles[0]))
            ul = max(df_train[c].quantile(xlim_quantiles[1]), df_test[c].quantile(xlim_quantiles[1]))
            plt.xlim([ll,ul])
            plt.legend(["Train", "Test"])
            plt.xlabel(str(c))
            plt.ylabel("Probability density")
        elif not (np.issubdtype(df_train[c], np.number) or np.issubdtype(df_test[c], np.number)):
            plot_df_train = pd.DataFrame(df_train[c].value_counts()/len(df_train[c]))
            plot_df_train["Data frame"] = "Train"
            plot_df_test = pd.DataFrame(df_test[c].value_counts()/len(df_test[c]))
            plot_df_test["Data frame"] = "Test"
            plot_df = pd.concat([plot_df_train, plot_df_test])
            if len(plot_df.index.unique()) < max_categories:
                ax = sns.barplot(x=plot_df.index, y=c, hue="Data frame", data=plot_df)
                ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
            else:
                train_levels = df_train[c].unique()
                test_levels = df_test[c].unique()
                ax = sns.barplot(
                    y=["Only train", "In both", "Only test"],
                    x=[
                        len([t for t in train_levels if t not in test_levels]),
                        len([t for t in train_levels if t in test_levels]),
                        len([t for t in test_levels if t not in train_levels])
                    ]
                )
                ax.set_yticklabels(ax.get_yticklabels(),rotation=30)
                plt.xlabel("Number of levels")
        else:
            plt.imshow([[[0.33,0.0,0.0]]])
            plt.text(-0.21,0.05,"Types of dataframe\ncolumns don't agree.\nWhat are you doing?")
    plt.tight_layout()


def index_value_plot(df, test_df = None, columns = None, target = None, subfigsize = (5,5), dpi=150, verbose=False):
    if columns is None:
        columns = [c for c in df.columns if np.issubdtype(df[c], np.number)]
    else:
        columns = [c for c in columns if np.issubdtype(df[c], np.number)]
    if len(columns) < 1:
        raise ValueError("No numeric features to plot.")
    subplot_cols = int(np.ceil(sqrt(len(columns))))
    subplot_rows = int(np.ceil(len(columns)/subplot_cols))
    fig = plt.figure(dpi=dpi, figsize=(subfigsize[0]*subplot_cols, subfigsize[1]*subplot_rows))
    for i, c in enumerate(columns):
        if verbose:
            print("column %d : %s" % (i, c))
        plt.subplot(subplot_rows, subplot_cols, i+1)
        if test_df is not None:
            sns.scatterplot(x=test_df[c], y=test_df.index,
                color='gray', alpha=1.0/3.0, linewidth=0)
        sns.scatterplot(
            x=df[c], y=df.index,
            hue=(df[target] if target is not None else None),
            alpha=1.0/3.0, linewidth=0)
        plt.title(c)
    plt.tight_layout()
    return fig
