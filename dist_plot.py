#!/usr/bin/env python

import sys
import json

import matplotlib
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt

from experiment import algorithms


def plot(algorithm):

    with open(algorithm + "/" + algorithm + "_brute_force.json", 'r') as fp:
        data = json.load(fp)
        bf_configs = [algorithms[algorithm]['total_ops'] / (d['time']/1e3) for d in data]

    maxp = max(bf_configs)
    minp = min(bf_configs)
    print('maxp', maxp)
    print('minp', minp)

    print('size', len(bf_configs))
    top = sorted(bf_configs)[-30:]
    print('top30', top)



    f, ax = plt.subplots(figsize=(13, 6))

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on")

    plt.xlabel("Performance in " + algorithms[algorithm]['unit'])


    plt.hist(bf_configs, bins=100, normed=True, color="#8BAFC8", linewidth=0, edgecolor="#757575")


    def to_percent(y, position):
        # Ignore the passed in position. This has the effect of scaling the default
        # tick locations.
        s = str(100 * y)

        # The percent symbol needs escaping in latex
        if matplotlib.rcParams['text.usetex'] is True:
            return s + r'$\%$'
        else:
            return s + '%'

    formatter = FuncFormatter(to_percent)

    # Set the formatter
    plt.gca().yaxis.set_major_formatter(formatter)

    f.savefig(algorithm + "-dist" + ".png", dpi=600, format='png')

    plt.show()



if __name__ == "__main__":
    if sys.argv[1] in algorithms:
        plot(sys.argv[1])
