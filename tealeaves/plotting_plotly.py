import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import scipy as sc
from math import sqrt
from scipy import stats as stats
from typing import Tuple
import networkx as nx
import string
import itertools as it
from sklearn.manifold import MDS
import dcor
import plotly.offline as py
import plotly.graph_objs as go

# Local imports
from tealeaves.util.histogram_bin_formulas import bin_it


def dataframe_plot(
    df,
    extra_test_dict={},
    quantiles=(0.025, 0.975),
    rare_category_factor=0.1,
    rows_per_pixel=1,
    show_value_ranks=True,
    fig=None,
):
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
    colours = {
        "na": {"rgb": "rgb(255.0, 0.0, 0.0)"},
        "-1": {"rgb": "rgb(255.0, 200.0, 0.0)"},
        "rare": {"rgb": "rgb(127.0, 0.0, 255.0)"},
        "empty_string": {"rgb": "rgb(255.0, 128.0, 0.0)"},
        "error": {"rgb": "rgb(128.0, 128.0, 128.0)"},
    }
    for i, key in enumerate(extra_test_dict.keys()):
        colours["extra_%d" % i] = {"rgb": extra_test_dict[key]}
    for i, key in enumerate(colours.keys()):
        colours[key]["value"] = 0.5 + i * 0.5 / len(colours)
    colourscale = [[0.0, "rgb(0.0,0.0,0.0)"], [0.5, "rgb(255.0,255.0,255.0)"]]
    for key in colours.keys():
        colourscale.append([colours[key]["value"], colours[key]["rgb"]])
        colourscale.append(
            [colours[key]["value"] + 0.5 / len(colours), colours[key]["rgb"]]
        )

    print(colourscale)

    img = np.ones((df.shape[0], df.shape[1])) * 1 / 2
    img[df.isna()] = colours["na"]["value"]
    for i, c in enumerate(df.columns):
        if len(df[c].unique()) < np.ceil(df.shape[0] / 10):
            counts = df[c].value_counts()
            img[
                df[c].isin(
                    counts[
                        counts < rare_category_factor * df.shape[0] / counts.shape[0]
                    ].index
                ),
                i,
            ] = colours["rare"]["value"]
        if is_numeric_dtype(df[c]):
            if show_value_ranks:
                ranks = stats.rankdata(df[c].values)
                img[:, i] = 0.5 * (
                    2 / 5
                    + (1 / 5) * (ranks - ranks.min()) / (ranks.max() - ranks.min())
                )
            img[df[c] < df[c].quantile(quantiles[0]), i] = 0
            img[df[c] > df[c].quantile(quantiles[1]), i] = 0.5
            img[df[c] == -1, i] = colours["-1"]["value"]
    img[df == ""] = colours["empty_string"]["value"]

    for i, key in enumerate(extra_test_dict.keys()):
        img[df == key] = colours["extra_%d" % i]["value"]

    # heatmap = go.Heatmap(
    #    z=img, y=df.index, x=df.columns, colorscale=colourscale, zmin=0, zmax=1
    # )

    py.iplot(
        [{"z": img, "type": "heatmap", "colorscale": colourscale, "zmin": 0, "zmax": 1}]
    )
    # patches = [
    #    mpatches.Patch(color=large_colour, label="Large value"),
    #    mpatches.Patch(color=small_colour, label="Small value"),
    #    mpatches.Patch(color=rare_colour, label="Rare value"),
    #    mpatches.Patch(color=na_colour, label="N/A"),
    #    mpatches.Patch(color=neg1_colour, label="-1"),
    #    mpatches.Patch(color=emptystr_colour, label='""'),
    # ]
    # for label, color in extra_test_dict.items():
    #    patches.append(mpatches.Patch(color=color, label=label))
    # plt.legend(handles=patches, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.0)
    # plt.tight_layout()


def dist_compare_plot(seq1, seq2, bins=None, max_bar_categories: int = 40):
    plot_object = None
    if (
        is_numeric_dtype(seq1)
        and is_numeric_dtype(seq2)
        and len(np.unique(np.concatenate((seq1, seq2)))) > 4
    ):
        if bins is None:
            bins = bin_it(np.concatenate((seq1.values, seq2.values), axis=0))
        frequencies1, edges1 = np.histogram(seq1.dropna(), bins, density=True)
        frequencies2, edges2 = np.histogram(seq2.dropna(), bins, density=True)
        plot_object = hv.NdOverlay(
            {
                "df1": hv.Histogram((edges1, frequencies1)),
                "df2": hv.Histogram((edges2, frequencies2)),
            }
        )
        plot_object.opts("Histogram", fill_alpha=0.5).redim.label(x="Value")
    else:
        plot_df_1 = pd.DataFrame(seq1.value_counts() / len(seq1))
        plot_df_1["Data frame"] = "df1"
        plot_df_2 = pd.DataFrame(seq2.value_counts() / len(seq2))
        plot_df_2["Data frame"] = "df2"
        plot_df = pd.concat([plot_df_1, plot_df_2]).reset_index()
        plot_df.columns = ["Category", "Count", "Source"]
        if len(plot_df.Category.unique()) < max_bar_categories:
            plot_object = hv.Bars(plot_df, ["Category", "Source"], "Count")
        else:
            train_levels = plot_df_1.index.unique()
            test_levels = plot_df_2.index.unique()
            plot_object = hv.Bars(
                (
                    ["Only df1", "In both", "Only df2"],
                    [
                        len([t for t in train_levels if t not in test_levels]),
                        len([t for t in train_levels if t in test_levels]),
                        len([t for t in test_levels if t not in train_levels]),
                    ],
                )
            ).opts(invert_axes=True)
    return plot_object


def dist_compare_grid(
    df1,
    df2,
    columns=None,
    max_bar_categories: int = 40,
    grid_size: Tuple[int, int] = (900, 900),
):
    if columns is None:
        columns = [c for c in df1.columns if c in df2.columns]
    else:
        for c in columns:
            if c not in df1.columns:
                raise ValueError("%s is not in df1.columns" % str(c))
            if c not in df2.columns:
                raise ValueError("%s is not in df2.columns" % str(c))
    grid_cols = int(np.ceil(np.sqrt(len(columns))))
    grid_rows = int(np.ceil(len(columns) / grid_cols))
    plots = [
        dist_compare_plot(df1[c], df2[c], max_bar_categories).opts(title=c)
        for c in columns
    ]
    grid = hv.Layout(plots).opts(shared_axes=False, normalize=False)
    grid.cols(grid_cols)
    # set sizes
    subplot_size = (int(grid_size[0] / grid_cols), int(grid_size[1] / grid_rows))
    grid.opts(
        opts.Histogram(width=subplot_size[0], height=subplot_size[1]),
        opts.Bars(width=subplot_size[0], height=subplot_size[1]),
    )
    return grid


# def index_value_plot(df, test_df = None, columns = None, target = None, subfigsize = (5,5), dpi=150, verbose=False):
#    if columns is None:
#        columns = [c for c in df.columns if np.issubdtype(df[c], np.number)]
#    else:
#        columns = [c for c in columns if np.issubdtype(df[c], np.number)]
#    if len(columns) < 1:
#        raise ValueError("No numeric features to plot.")
#    subplot_cols = int(np.ceil(sqrt(len(columns))))
#    subplot_rows = int(np.ceil(len(columns)/subplot_cols))
#    fig = plt.figure(dpi=dpi, figsize=(subfigsize[0]*subplot_cols, subfigsize[1]*subplot_rows))
#    for i, c in enumerate(columns):
#        if verbose:
#            print("column %d : %s" % (i, c))
#        plt.subplot(subplot_rows, subplot_cols, i+1)
#        if test_df is not None:
#            sns.scatterplot(x=test_df[c], y=test_df.index,
#                color='gray', alpha=1.0/3.0, linewidth=0)
#        sns.scatterplot(
#            x=df[c], y=df.index,
#            hue=(df[target] if target is not None else None),
#            alpha=1.0/3.0, linewidth=0)
#        plt.title(c)
#    plt.tight_layout()
#    return fig


def relation_graph(df, distance="dcorr", min_corr=None, iterations=20):
    if distance in ("dcorr", "distance_correlation"):
        numeric = [c for c in df.columns if is_numeric_dtype(df[c])]
        corr = pd.DataFrame(
            data=np.diag(np.ones(len(numeric))), columns=numeric, index=numeric
        )
        dist = pd.DataFrame(
            data=np.diag(np.ones(len(numeric))), columns=numeric, index=numeric
        )
        for a, b in it.combinations(numeric, 2):
            c = dcor.distance_correlation_af_inv(df[[a]].values, df[[b]].values)
            corr.loc[a, b] = c
            corr.loc[b, a] = c
        dist = 1 - corr
        if min_corr is None:
            min_corr = 0.2
    elif distance in ("corr", "correlation"):
        corr = df.corr()
        dist = 1 - corr.abs()
        if min_corr is None:
            min_corr = 0.2
    else:
        raise ValueError("Distance %s is not implemented" % distance)

    G = nx.Graph()
    for n in corr.columns:
        G.add_node(n)

    for u, v in it.combinations(corr.columns, 2):
        if abs(corr.loc[u, v]) > min_corr:
            G.add_edge(u, v, corr=corr.loc[u, v])

    labels = {n: n for n in G.nodes()}
    edge_labels = {e: "%.2f" % G.edges[e]["corr"] for e in G.edges()}

    pos = MDS(
        dissimilarity="precomputed", metric=True, n_init=iterations
    ).fit_transform(dist)
    pos = np.array(pos)
    node_pos = {n: p for n, p in zip(dist.columns, pos)}

    nx.draw_networkx_nodes(G, node_pos, node_color="grey", node_size=150)
    nx.draw_networkx_edges(
        G,
        node_pos,
        edgelist=[e for e in G.edges() if G.edges[e]["corr"] >= 0],
        edge_color="steelblue",
        alpha=1.0,
    )
    nx.draw_networkx_edges(
        G,
        node_pos,
        edgelist=[e for e in G.edges() if G.edges[e]["corr"] < 0],
        edge_color="indianred",
        alpha=1.0,
    )
    nx.draw_networkx_edge_labels(
        G, node_pos, edge_labels=edge_labels, font_family="monospace"
    )

    x_scale = np.max(pos[:, 1]) - np.min(pos[:, 1])
    for c in labels:
        x, y = node_pos[c]
        plt.text(
            x,
            y + 0.295 * x_scale / plt.gcf().get_size_inches()[1],
            s=c,
            bbox=dict(facecolor="lightblue", alpha=0.2),
            fontdict=dict(),
            horizontalalignment="center",
        )

    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
