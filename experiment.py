#!/usr/bin/env python

import gc
import os
import sys
import time
import json
from collections import OrderedDict

try:
    import pycuda.driver as drv
    pycuda_available = True
except:
    drv = None
    pycuda_available = False

import numpy
import logging
import kernel_tuner
from kernel_tuner.util import get_config_string

strategy_options = OrderedDict()
strategy_options["minimize"] = ["Nelder-Mead", "Powell", "CG", "BFGS",
                                        "L-BFGS-B", "TNC", "COBYLA", "SLSQP"]
strategy_options["basinhopping"] = ["Nelder-Mead", "Powell", "CG", "BFGS",
                                        "L-BFGS-B", "TNC", "COBYLA", "SLSQP"]
strategy_options["diff_evo"] = ["best1bin", "best1exp", "rand1exp", "randtobest1exp", "best2exp",
                                    "rand2exp", "randtobest1bin", "best2bin", "rand2bin", "rand1bin"]
strategy_options["brute_force"] = [None]
strategy_options["genetic_algorithm"] = [None]
strategy_options["pso"] = [None]
strategy_options["simulated_annealing"] = [None]
strategy_options["firefly_algorithm"] = [None]


from convolution.convolution import tune as tune_convolution
from gemm.xgemm import tune as tune_gemm
from gemm_amd.xgemm import tune as tune_gemm_amd

if pycuda_available:
    from convolution_streams.convolution_streams import tune as tune_convolution_streams
    from pnpoly.pnpoly import tune as tune_pnpoly
else:
    tune_convolution_streams = None
    tune_pnpoly = None



algorithms = {}
algorithms["convolution"]  = {"method": tune_convolution,
                              "total_ops": (4096*4096*17*17*2)/1e9,
                              "unit": "GFLOP/s",
                              "device": "GTX Titan X (Maxwell)",
                              "fancy_name": "2D Convolution"}
algorithms["pnpoly"]       = {"method": tune_pnpoly,
                              "total_ops": 2e1,
                              "unit": "MPoints/s",
                              "device": "GTX Titan X (Maxwell)",
                              "fancy_name": "Point-in-Polygon"}
algorithms["gemm"]         = {"method": tune_gemm,
                              "total_ops": (2.0 * (2048**3) + 2.0 * 2048 * 2048)/1e9,
                              "unit": "GFLOP/s",
                              "device": "GTX Titan X (Maxwell)",
                              "fancy_name": "GEMM"}
algorithms["convolution_streams"]  = {"method": tune_convolution_streams,
                              "total_ops": (4096*4096*17*17*2)/1e9,
                              "unit": "GFLOP/s",
                              "device": "GTX Titan X (Maxwell)",
                              "fancy_name": "2D Convolution with streams"}
algorithms["gemm_amd"]     = {"method": tune_gemm_amd,
                              "total_ops": (2.0 * (2048**3) + 2.0 * 2048 * 2048)/1e9,
                              "unit": "GFLOP/s",
                              "device": "AMD Vega",
                              "fancy_name": "GEMM"}


needs_context = ["pnpoly", "convolution_streams"]





def tune(algorithm, do_strategy):

    result_summary = {}

    tune_func = algorithms[algorithm]['method']

    test_methods = strategy_options[do_strategy]

    for method in test_methods:

        if method:
            experiment_name = do_strategy + "_" + method
        else:
            experiment_name = do_strategy

        summary = OrderedDict()
        summary["best"] = []
        summary["best_times"] = []
        summary["execution_time"] = []

        try:

            #test all methods multiple times because some methods are stochastic
            for i in range(32 if do_strategy != "brute_force" else 1):

                outfile = algorithm + "/" + algorithm + "_" + experiment_name
                if do_strategy != "brute_force":
                    outfile += "_" + str(i)
                outfile += ".json"

                if os.path.isfile(outfile):
                    print("output file %s already exists, skipping this experiment" % outfile)
                    continue

                start = time.time()

                if 'options' in algorithms[algorithm]:
                    results, env = tune_func(do_strategy, method, algorithms[algorithm]['options'])
                else:
                    results, env = tune_func(do_strategy, method)

                end = time.time()
                env['execution_time'] = end-start

                gc.collect()

                with open(outfile, 'w') as fp:
                    json.dump(results, fp)

                best_config = min(results, key=lambda x:x['time'])
                summary["best"].append(best_config)
                summary["best_times"].append(best_config['time'])
                summary["execution_time"].append(env['execution_time'])

        finally:
            if len(summary["best"]) > 0:
                result_summary[experiment_name] = summary
                update_results_db(algorithm, result_summary)



    #print some output at end of run, not strictly necessary
    with open(algorithm + "/" + algorithm + "_summary.json", 'r') as fp:
        result_summary = json.load(fp)

    total_ops = algorithms[algorithm]['total_ops']
    unit = algorithms[algorithm]['unit']

    for k, d in result_summary.items():
        print(k)
        for i, config in enumerate(d["best"]):
            print(get_config_string(config), str(total_ops / (config['time'] /1e3)) + " " + unit, str(d["execution_time"][i]) + " sec")
        print("average best performance: " + str(numpy.average(d["best_times"])))
        print("average execution_time: " + str(numpy.average(d["execution_time"])))




def update_results_db(algorithm, result_summary):

    #update result summary database
    #first read in the db that is there
    results_db = {}
    try:
        with open(algorithm + "/" + algorithm + "_summary.json", 'r') as fp:
            results_db = json.load(fp)
    except:
        pass

    #merge old and new db
    for k in result_summary.keys():
        #append new results to existing db records
        if k in results_db.keys():
            for sub_k, sub_v in result_summary[k].items():
                results_db[k][sub_k].extend(sub_v)
        #add to db
        else:
            results_db[k] = result_summary[k]

    #store updated db
    with open(algorithm + "/" + algorithm + "_summary.json", 'w') as fp:
        json.dump(results_db, fp)

    result_summary = {}





if __name__ == "__main__":
    """use as ./experiment.py [algorithm] [strategy]"""

    arg1 = sys.argv[1]
    if not arg1 in algorithms:
        raise ValueError("unknown algorithm")

    arg2 = sys.argv[2]
    if not arg2 in strategy_options:
        raise ValueError("unknown strategy")

    #algorithm specific cleanup

    context = None
    if arg1 in needs_context and pycuda_available:
        drv.init()
        context = drv.Device(0).make_context()

    try:
        tune(arg1, arg2)
    finally:
        if arg1 in needs_context and pycuda_available:
            context.pop()


