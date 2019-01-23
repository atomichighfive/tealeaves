import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from os.path import realpath, split

# Import functions to demo
from tealeaves.plotting import dataframe_plot, index_value_plot, dist_compare_plot

def load_demo_data():
    games = pd.read_csv('/'.join([split(realpath(__file__))[0], 'demos','sample_data','titanic.csv']))
    train = games.iloc[1*int(len(games)/3):len(games)].reset_index(drop=True)
    test = games.iloc[0:1*int(len(games)/3)].reset_index(drop=True)
    return games, train, test

def demo_dist_compare_plot():
    games, train, test = load_demo_data()
    dist_compare_plot(
        train,
        test,
        histogram=True,
        kde=True
    )
    plt.savefig('/'.join([split(realpath(__file__))[0], 'demos','sample_images','demo_dist_compare_plot.png']))


def demo_index_value_plot():
    games, train, test = load_demo_data()
    index_value_plot(
        train,
        test,
        target='Survived')
    plt.savefig('/'.join([split(realpath(__file__))[0], 'demos','sample_images','demo_index_value_plot.png']))


def demo_dataframe_plot():
    games, train, test = load_demo_data()
    dataframe_plot(
        games,
        rows_per_pixel=1,
        extra_test_dict={13:np.array([0,0,255])/255}
    )
    plt.savefig('/'.join([split(realpath(__file__))[0], 'demos','sample_images','demo_dataframe_plot.png']))
