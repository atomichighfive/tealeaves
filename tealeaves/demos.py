import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Import functions to demo
from plotting import dataframe_plot, index_value_plot, dist_compare_plot

def load_demo_data():
    games = pd.read_csv('./demos/sample_data/video-game-sales-with-ratings.zip')
    games.loc[np.random.choice(np.arange(0,games.shape[0],1), int(games.shape[0]*0.05), replace=False), 'Platform'] = ""
    games.loc[np.random.choice(np.arange(0,games.shape[0],1), int(games.shape[0]*0.05), replace=False), 'Global_Sales'] = -1
    games["User_Score"] = games["User_Score"].astype(np.float32, errors='ignore')
    train = games.iloc[1*int(len(games)/3):len(games)].reset_index(drop=True)
    test = games.iloc[0:1*int(len(games)/3)].reset_index(drop=True)
    return all, train, test

def demo_dist_compare_plot():
    games, train, test = load_demo_data()
    dist_compare_plot(
        train,
        test,
        histogram=True,
        kde=True
    )
    plt.savefig("./demos/sample_images/demo_dist_compare_plot.png")


def demo_index_value_plot():
    games, train, test = load_demo_data()
    index_value_plot(
        train,
        test,
        target='Rating')
    plt.savefig("./demos/sample_images/demo_index_value_plot.png")


def demo_dataframe_plot():
    games, train, test = load_demo_data()
    dataframe_plot(
        games,
        rows_per_pixel=15,
        extra_test_dict={"PS3":np.array([0,0,255])/255}
    )
    plt.savefig("./demos/sample_images/demo_dataframe_plot.png")
