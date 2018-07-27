#!/usr/bin/env python

import os
import json
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from matplotlib.lines import Line2D

from experiment import strategy_options, algorithms


abbreviate_methods = OrderedDict()
abbreviate_methods["brute_force"] = "bf"
abbreviate_methods["minimize"] = "M-"
abbreviate_methods["basinhopping"] = "BS-"
abbreviate_methods["diff_evo"] = "DE-"
abbreviate_methods["genetic"] = "GA"
abbreviate_methods["firefly"] = "FA"
abbreviate_methods["pso"] = "PSO"
abbreviate_methods["annealing"] = "SA"


brute_force_data = {}
#brute_force_data['convolution'] = dict(block_size_x=128, block_size_y=2, tile_size_x=1, tile_size_y=4, read_only=1, time=2.55226883888, perf=3799.45509668, execution_time=3629.2007822990417)
#brute_force_data['pnpoly'] = dict(block_size_x=896, tile_size=20, between_method=0, use_precomputed_slopes=1, use_method=2, time=26.7275260925, perf=748292226.178, execution_time=14093.25716304779)

def append_flat(the_list, whatever):
    if hasattr(whatever, '__iter__'):
        the_list += list(whatever)
    else:
        the_list += list([whatever])



def make_plots(algorithm):

    summary = {}
    all_points = {}

    with open(algorithm + "/" + algorithm + "_summary.json", 'r') as fp:
        data = json.load(fp)

    #see if brute_force is there
    bf_data = data.pop('brute_force', None)

    if bf_data:
        summary['brute_force'] = {}
        shortname = 'brute_force'
        summary[shortname]['execution_time'] = np.average(bf_data['execution_time'])
        summary[shortname]['execution_time_err'] = np.std(bf_data['execution_time'])

        best_performances = algorithms[algorithm]['total_ops'] / (np.array(bf_data['best_times']) / 1e3)
        summary[shortname]['best_perf'] = np.average(best_performances)
        summary[shortname]['max_perf'] = summary[shortname]['best_perf']
        summary[shortname]['best_perf_err'] = np.std(best_performances)
        summary[shortname]['best_conf'] = {}
    else:
        bf_data = brute_force_data.get(algorithm, None)
        summary['brute_force'] = {}
        summary['brute_force']['execution_time'] = bf_data['execution_time']
        summary['brute_force']['execution_time_err'] = 0.0
        summary['brute_force']['best_perf'] = algorithms[algorithm]['total_ops'] / (bf_data['time'] / 1e3)
        summary['brute_force']['max_perf'] = summary['brute_force']['best_perf']
        summary['brute_force']['best_perf_err'] = 0.0
        summary['brute_force']['best_conf'] = {}



    #read the data
    for k,v in data.items():
        if "_" in k:
            strat = "_".join(k.split("_")[:-1])
            method = "".join(k.split("_")[-1])
        else:
            strat = k
            method = ""

        if strat == "simulated":
            strat = "annealing"
        shortname = abbreviate_methods[strat] + method

        print(shortname)

        summary[shortname] = {}
        all_points[shortname] = {}
        #plot only average
        print(v['execution_time'])
        summary[shortname]['execution_time'] = np.average(v['execution_time'])
        summary[shortname]['execution_time_err'] = np.std(v['execution_time'])
        #plot all measurements
        all_points[shortname]['execution_time'] = v['execution_time']
        #summary[shortname]['execution_time_err'] = v['execution_time']

        best_performances = algorithms[algorithm]['total_ops'] / (np.array(v['best_times']) / 1e3)
        print(best_performances)

        summary[shortname]['max_perf'] = np.amax(best_performances)
        summary[shortname]['best_conf'] = v['best'][np.argmax(best_performances)]

        summary[shortname]['best_perf'] = np.average(best_performances)
        summary[shortname]['best_perf_err'] = np.std(best_performances)
        #plot all measurements
        all_points[shortname]['best_perf'] = best_performances
        #summary[shortname]['best_perf_err'] = best_performances

    method_names = sorted(summary.keys())

    print('method', 'best', 'best_err', 'time', 'time_err')
    for k in method_names:
        print("%s & %.2f & %.2f & %.2f & %.2f \\\\" % (k, summary[k]['best_perf'], summary[k]['best_perf_err'], summary[k]['execution_time'], summary[k]['execution_time_err']))
        #print(k, summary[k]['best_perf'], summary[k]['execution_time'])


    #maxp = max([v['best_perf'] for v in summary.values()])
    all_perfs = []
    for v in summary.values():
        append_flat(all_perfs, v['best_perf'])
    print(all_perfs)
    maxp = np.amax(all_perfs)
    print('maxp', maxp)

    plot(algorithm, summary, all_points, method_names, maxp)



def plot(algorithm, summary, all_points, method_names, maxp):

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 6))

    #y = [v['execution_time'] for v in summary.values()]
    #x = [v['best_perf'] for v in summary.values()]
    #yerr = [v['execution_time_err'] for v in summary.values()]
    #xerr = [v['best_perf_err'] for v in summary.values()]

    color_names = OrderedDict()
    color_names['b'] = '#949494'
    color_names['M'] = '#ECE133'
    color_names['B'] = '#D55E00'
    color_names['D'] = '#029E73'
    color_names['G'] = '#0173B2'
    color_names['F'] = '#56B4E9'
    color_names['P'] = '#F14CC1'
    color_names['S'] = '#FBAFE4'

    all_x = []
    all_y = []
    all_colors = []
    for k,v in all_points.items():
        append_flat(all_y, v['execution_time'])
        append_flat(all_x, v['best_perf'])
        all_colors += [color_names[k[0]] for _ in v['best_perf']]

    y = []
    x = []
    yerr = []
    xerr = []
    colors = []
    methods = []

    for k,v in summary.items():
        append_flat(y, v['execution_time'])
        append_flat(x, v['best_perf'])
        append_flat(yerr, v['execution_time_err'])
        append_flat(xerr, v['best_perf_err'])
        try:
            colors += [color_names[k[0]] for _ in v['best_perf']]
        except:
            colors += [color_names[k[0]]]
        methods.append(k)

    #plt.errorbar(x, y, xerr=xerr, fmt='o')
    #plt.errorbar(x, y, xerr=xerr, fmt='o')
    plt.scatter(all_x, all_y, alpha=0.05, c=all_colors, linewidths=0.0, s=100.0)
    plt.scatter(x, y, alpha=1.0, c=colors, linewidths=0.2, s=100.0)


    #build legend
    lines = [Line2D(range(1), range(1), color="white", marker='o', markersize=10.0, markeredgewidth=0.2, markerfacecolor=c) for c in color_names.values()]
    plt.legend(lines,list(abbreviate_methods.keys()),numpoints=1, loc=2, framealpha=0.5)


    f.suptitle('Auto-tuning ' + algorithms[algorithm]['fancy_name'] + ' on ' + algorithms[algorithm]['device'])

    plt.xlabel(algorithms[algorithm]['unit'])
    plt.ylabel('time (s)')

    #plt.xticks(rotation=35)
    #ax.set_xticklabels(method_names)

    # Finalize the figure
    #sns.despine(left=True, bottom=True)
    ax.set_yscale("log")

    if maxp > 1000:
        ax.set(xlim=(0, (maxp//500+1)*500 ), ylim=(0,None))
    else:
        ax.set(xlim=(0, (maxp//100+1)*100 ), ylim=(0,None))


    ax.set_axisbelow(True)
    ax.yaxis.grid(which="both", linestyle='-', alpha=0.2, linewidth=0.5)


    #f.subplots_adjust(bottom=0.175, right=0.98, left=0.1, wspace=0.0, hspace=0.0)
    #f.tight_layout()

    filename = (algorithm + "-" + 'summary').replace("_", "-")
    f.savefig(filename + ".pdf", format='pdf')
    f.savefig(filename + ".png", dpi=600, format='png')
    plt.show()






if __name__ == "__main__":

    matplotlib.rcParams.update({'font.size': 16})

    if len(sys.argv) < 2 or sys.argv[1] == "-h":
        print("Usage: ./scatter.py [algorithm]")
        exit()

    arg1 = sys.argv[1]
    if not arg1 in algorithms:
        raise ValueError("unknown algorithm")

    make_plots(arg1)
