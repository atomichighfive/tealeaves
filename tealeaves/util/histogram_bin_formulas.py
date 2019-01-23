# Author: Eyad Sibai
import numpy as np
from scipy.stats import iqr

def histogram_square_formula(values):
    """ reference https://www.kdnuggets.com/2018/02/histogram-tips-tricks.html
    :param values: the values of the histogram data
    :return: number of bins and width of the bin
    """
    bins = np.sqrt(len(values))
    bin_width = (np.max(values) - np.min(values)) / bins
    return bins, bin_width


def histogram_sturges_formula(values):
    """
    :param values:
    :return:
    """
    bins = np.ceil(np.log2(len(values))) + 1
    bin_width = (np.max(values) - np.min(values)) / bins
    return bins, bin_width


def histogram_rice_formula(values):
    """
    :param values:
    :return:
    """
    bins = 2 * np.power(len(values), 1. / 3)
    bin_width = (np.max(values) - np.min(values)) / bins
    return bins, bin_width


def histogram_scott_formula(values):
    """
    :param values:
    :return:
    """
    bin_width = 3.5 * np.std(values) / np.power(len(values), 1 / 3)
    bins = (np.max(values) - np.min(values)) / bin_width
    return bins, bin_width


def histogram_freedman_diaconis_formula(values):
    """
    :param values:
    :return:
    """
    bin_width = 2 * iqr(values) / np.power(len(values), 1 / 3)
    bins = (np.max(values) - np.min(values)) / bin_width
    return bins, bin_width


# Author: Andreas Syrén
def bin_it(values, formula='freedman_diaconis'):
    values = [v for v in values if np.isfinite(v)]
    if formula == 'square':
        bins, bin_width = histogram_square_formula(values)
    elif formula == 'sturges':
        bins, bin_width = histogram_sturges_formula(values)
    elif formula == 'rice':
        bins, bin_width = histogram_rice_formula(values)
    elif formula == 'scott':
        bins, bin_width = histogram_scott_formula(values)
    elif ('freedman' in formula) or ('diaconis' in formula):
        bins, bin_width = histogram_freedman_diaconis_formula(values)
    else:
        raise ValueError("\"%s\" not in implemented formulas." % str(formula))
    return np.linspace(np.min(values), np.max(values), min(500, bins))
