import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from os.path import realpath, split

# Import functions to demo
from tealeaves.plotting import dataframe_plot, dist_compare_grid, relation_graph


def load_demo_data():
    titanic = pd.read_csv(
        "/".join([split(realpath(__file__))[0], "demos", "sample_data", "titanic.csv"])
    )
    train = titanic.iloc[1 * int(len(titanic) / 3) : len(titanic)].reset_index(
        drop=True
    )
    test = titanic.iloc[0 : 1 * int(len(titanic) / 3)].reset_index(drop=True)
    return titanic, train, test


def demo_dist_compare_grid():
    titanic, train, test = load_demo_data()
    return dist_compare_grid(train, test, grid_size=(950, 712.5))


# def demo_index_value_plot():
#    titanic, train, test = load_demo_data()
#    index_value_plot(
#        train,
#        test,
#        target='Survived'
#    )


def demo_dataframe_plot():
    titanic, train, test = load_demo_data()
    dataframe_plot(
        titanic, rows_per_pixel=1, extra_test_dict={13: np.array([0, 0, 255]) / 255}
    )


def demo_relation_graph():
    titanic, train, test = load_demo_data()
    relation_graph(titanic, distance="dcorr", min_corr=None, iterations=20)
