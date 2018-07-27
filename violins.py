#!/usr/bin/env python

import os
import json
import sys

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from experiment import strategy_options, algorithms

plots = {}
plots["minimize"] = ["minimize"]
plots["basinhopping"] = ["basinhopping"]
plots["diff_evo"] = ["diff_evo"]
plots["other"] = ["genetic_algorithm", "pso", "firefly_algorithm", "simulated_annealing"]





def make_plots(algorithm):


    method_names = []

    abbreviate_methods = {}
    abbreviate_methods["minimize"] = "M-"
    abbreviate_methods["basinhopping"] = "BS-"
    abbreviate_methods["diff_evo"] = "DE-"
    abbreviate_methods["genetic_algorithm"] = "GA"
    abbreviate_methods["firefly_algorithm"] = "FA"
    abbreviate_methods["pso"] = "PSO"
    abbreviate_methods["simulated_annealing"] = "SA"

    with open(algorithm + "/" + algorithm + "_brute_force.json", 'r') as fp:
        data = json.load(fp)
        bf_configs = [algorithms[algorithm]['total_ops'] / (d['time']/1e3) for d in data]

    maxp = max(bf_configs)
    minp = min(bf_configs)
    print('maxp', maxp)


    for this_plot,strats in plots.items():

    #for strat in [s for s in strategy_options if not s == "brute_force"]:

        collect_data = []
        method_names = []

        for strat in strats:

            for method in strategy_options[strat]:

                filename = algorithm + "/" + algorithm + "_" + strat + "_"
                if method:
                    filename += method + "_"

                if not os.path.isfile(filename + str(0) + ".json"):
                    continue

                data_per_method = []

                for i in range(32):

                    if not os.path.isfile(filename + str(i) + ".json"):
                        continue

                    with open(filename + str(i) + ".json", 'r') as fp:
                        data = json.load(fp)
                        configs = [algorithms[algorithm]['total_ops'] / (d['time']/1e3) for d in data]

                    data_per_method += configs

                collect_data.append(data_per_method)
                if strat in abbreviate_methods:
                    #method_names += [abbreviate_methods[strat] + method]
                    if method:
                        method_names += [method]
                    else:
                        method_names += [abbreviate_methods[strat]]
                else:
                    method_names += [strat]

        #also append brute_force for comparison
        collect_data.append(bf_configs)
        method_names += ["brute_force"]

        if len(collect_data) > 1:
            plot(algorithm, collect_data, method_names, this_plot, maxp)



def plot(algorithm, collect_data, method_names, strat, maxp):

    sns.set(style="whitegrid", font_scale=1.45)
    fancy_name = {"minimize": "Minimize", "basinhopping": "Basin Hopping", "diff_evo": "Differential Evolution", "genetic_algorithm": "Genetic Algorithm", "other": "Global Optimization methods"}

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 6))

    # Draw a violinplot with a narrower bandwidth than the default
    ax = sns.violinplot(data=collect_data, palette="Set3", cut=0, linewidth=2)

    f.suptitle('Auto-tuning ' + algorithms[algorithm]['fancy_name'] + ' with ' + fancy_name[strat] + ' on ' + algorithms[algorithm]['device'])

    plt.ylabel(algorithms[algorithm]['unit'])
    plt.xticks(rotation=25)
    ax.set_xticklabels(method_names)

    # Finalize the figure
    if maxp > 1000:
        ax.set(ylim=(0, (maxp//500+1)*500 ))
    else:
        ax.set(ylim=(0, (maxp//100+1)*100 ))
    sns.despine(left=True, bottom=True)

    f.subplots_adjust(bottom=0.19, right=0.98, left=0.1, wspace=0.0, hspace=0.0)
    #f.tight_layout()

    filename = (algorithm + "-" + strat).replace('_', '-')

    f.savefig(filename + ".pdf", format='pdf')
    f.savefig(filename + ".png", dpi=600, format='png')
    #plt.show()






if __name__ == "__main__":

    if len(sys.argv) < 2 or sys.argv[1] == "-h":
        print("Usage: ./violins.py [algorithm]")
        exit()

    arg1 = sys.argv[1]
    if not arg1 in algorithms:
        raise ValueError("unknown algorithm")

    make_plots(arg1)
