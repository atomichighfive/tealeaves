#%% Import libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import scipy as sc
import scipy.stats as stats

#%%
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

def demo_dataframe_plot():
    games = pd.read_csv('./sample_data/video-game-sales-with-ratings.zip')
    print(games.Platform.unique())
    games.loc[np.random.choice(np.arange(0,games.shape[0],1), int(games.shape[0]*0.05), replace=False), 'Platform'] = ""
    games.loc[np.random.choice(np.arange(0,games.shape[0],1), int(games.shape[0]*0.05), replace=False), 'Global_Sales'] = -1
    dataframe_plot(games, rows_per_pixel=15, extra_test_dict={"PS3":np.array([0,0,255])/255})
    plt.savefig("output/demo_dataframe_plot.png")

