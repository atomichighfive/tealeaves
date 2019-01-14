#%%
import pandas as pd
import numpy as np
import plotly as plt
import seaborn as sns
from math import sqrt
from matplotlib import pyplot as plt

#%%
def index_value_plot(df, columns = None, target = None, subfigsize = (10,10), dpi=150, verbose=False):
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
        sns.scatterplot(
            x=df[c], y=df.index, 
            hue=(df[target] if target is not None else None),
            alpha=1.0/3.0, linewidth=0)
        plt.title(c)
    plt.tight_layout()
    return fig

def demo_index_value_plot():
    games = pd.read_csv('./sample_data/video-game-sales-with-ratings.zip')
    games.loc[np.random.choice(np.arange(0,games.shape[0],1), int(games.shape[0]*0.05), replace=False), 'Platform'] = ""
    games.loc[np.random.choice(np.arange(0,games.shape[0],1), int(games.shape[0]*0.05), replace=False), 'Global_Sales'] = -1
    print(games.columns)
    index_value_plot(games, columns=['EU_Sales','JP_Sales','NA_Sales'], target='Platform')
    plt.savefig("output/demo_index_value_plot.png")

