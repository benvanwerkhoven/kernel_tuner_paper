#!/usr/bin/env python

from collections import OrderedDict
import os

import numpy
import logging
import kernel_tuner
from kernel_tuner.util import get_config_string

def tune(do_strategy, do_method):
    with open(os.path.dirname(os.path.realpath(__file__)) + '/convolution.cu', 'r') as f:
        kernel_string = f.read()

    #setup tunable parameters
    tune_params = OrderedDict()
    #tune_params["filter_height"] = [i for i in range(3,35,2)]
    #tune_params["filter_height"] = [17]
    #tune_params["filter_width"] = [i for i in range(3,35,2)]
    #tune_params["filter_width"] = [17]
    filter_width = 17
    filter_height = 17

    tune_params["block_size_x"] = [16*i for i in range(1,17)]
    tune_params["block_size_y"] = [2**i for i in range(5)]
    tune_params["tile_size_x"] = [i for i in range(1,5)]
    tune_params["tile_size_y"] = [i for i in range(1,5)]

    #tune_params["use_padding"] = [0,1]  #toggle the insertion of padding in shared memory
    tune_params["read_only"] = [0,1]    #toggle using the read-only cache

    #limit the search to only use padding when its effective
    #restrict = ["use_padding==0 or (block_size_x % 32 != 0)", "(block_size_x*tile_size_x+4)*(block_size_y*tile_size_y+4) < 12*1024"]
    restrict = ["((block_size_x*tile_size_x+%d)*(block_size_y*tile_size_y+%d)) < 12*1024" % (filter_width-1, filter_height-1)]

    print(restrict)

    problem_size = (4096, 4096)
    size = numpy.prod(problem_size)
    largest_fh = filter_height
    largest_fw = filter_width
    input_size = ((problem_size[0]+largest_fw-1) * (problem_size[1]+largest_fh-1))

    output_image = numpy.zeros(size).astype(numpy.float32)
    input_image = numpy.random.randn(input_size).astype(numpy.float32)
    filter_weights = numpy.random.randn(largest_fh * largest_fw).astype(numpy.float32)

    cmem_args = {'d_filter': filter_weights}
    args = [output_image, input_image, filter_weights]

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    #start tuning
    results, env = kernel_tuner.tune_kernel("convolution_kernel", kernel_string,
                    problem_size, args, tune_params,
                    grid_div_y=grid_div_y, grid_div_x=grid_div_x, cmem_args=cmem_args,
                    verbose=True, strategy=do_strategy, method=do_method, restrictions=restrict)

    return results, env


if __name__ == "__main__":
    tune(None, None)
