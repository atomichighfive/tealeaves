#%% Import libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns
import scipy.stats as stats
from math import sqrt
from histogram_bin_formulas import bin_it

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


def demo_dist_compare_plot():
    games = pd.read_csv('./sample_data/video-game-sales-with-ratings.zip')
    games.loc[np.random.choice(np.arange(0,games.shape[0],1), int(games.shape[0]*0.05), replace=False), 'Platform'] = ""
    games.loc[np.random.choice(np.arange(0,games.shape[0],1), int(games.shape[0]*0.05), replace=False), 'Global_Sales'] = -1
    games["User_Score"] = games["User_Score"].astype(np.float32, errors='ignore')
    print(games.columns)
    dist_compare_plot(
        games.iloc[1*int(len(games)/3):len(games)].reset_index(drop=True),
        games.iloc[0:1*int(len(games)/3)].reset_index(drop=True),
        histogram=True, kde=True
        )
    plt.savefig("output/demo_dist_compare_plot.png")
