#!/usr/bin/env python
""" Point-in-Polygon host/device code tuner

This program is used for auto-tuning the host and device code of a CUDA program
for computing the point-in-polygon problem for very large datasets and large
polygons.

The time measurements used as a basis for tuning include the time spent on
data transfers between host and device memory. The host code uses device mapped
host memory to overlap communication between host and device with kernel
execution on the GPU. Because each input is read only once and each output
is written only once, this implementation almost fully overlaps all
communication and the kernel execution time dominates the total execution time.

The code has the option to precompute all polygon line slopes on the CPU and
reuse those results on the GPU, instead of recomputing them on the GPU all
the time. The time spent on precomputing these values on the CPU is also
taken into account by the time measurement in the code.

This code was written for use with the Kernel Tuner. See:
     https://github.com/benvanwerkhoven/kernel_tuner

Author: Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
"""
import os
from collections import OrderedDict
import numpy
import kernel_tuner
import json
import logging

from kernel_tuner.util import get_config_string

import pycuda.driver as drv

def allocate(n, dtype=numpy.float32):
    """ allocate context-portable device mapped host memory """
    return drv.pagelocked_empty(int(n), dtype, order='C', mem_flags=drv.host_alloc_flags.PORTABLE|drv.host_alloc_flags.DEVICEMAP)


def tune(do_strategy, do_method):

    #set the number of points and the number of vertices
    size = numpy.int32(2e7)
    problem_size = (size, 1)
    vertices = 600

    #allocate device mapped host memory and generate input data
    points = allocate(2*size, numpy.float32)
    numpy.copyto(points, numpy.random.randn(2*size).astype(numpy.float32))

    bitmap = allocate(size, numpy.int32)
    numpy.copyto(bitmap, numpy.zeros(size).astype(numpy.int32))
    #as test input we use a circle with radius 1 as polygon and
    #a large set of normally distributed points around 0,0
    vertex_seeds = numpy.sort(numpy.random.rand(vertices)*2.0*numpy.pi)[::-1]
    vertex_x = numpy.cos(vertex_seeds)
    vertex_y = numpy.sin(vertex_seeds)
    vertex_xy = allocate(2*vertices, numpy.float32)
    numpy.copyto(vertex_xy, numpy.array( list(zip(vertex_x, vertex_y)) ).astype(numpy.float32).ravel())

    #kernel arguments
    args = [bitmap, points, vertex_xy, size]

    #setup tunable parameters
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,32)]  #multiple of 32
    tune_params["tile_size"] = [1] + [2*i for i in range(1,11)]
    tune_params["between_method"] = [0, 1, 2, 3]
    tune_params["use_precomputed_slopes"] = [0, 1]
    tune_params["use_method"] = [0, 1, 2]

    #tell the Kernel Tuner how to compute the grid dimensions from the problem_size
    grid_div_x = ["block_size_x", "tile_size"]

    path = os.path.dirname(os.path.realpath(__file__)) + '/'

    results, env = kernel_tuner.tune_kernel("cn_pnpoly_host", [path+'pnpoly_host.cu', path+'pnpoly.cu'],
            problem_size, args, tune_params,
            grid_div_x=grid_div_x, lang="C", compiler_options=["-O3", "-arch=sm_52", "-I"+path], verbose=True,
            strategy=do_strategy, method=do_method)

    return results, env




if __name__ == "__main__":
    drv.init()
    context = drv.Device(0).make_context()
    try:
        tune(None, None)
    finally:
        context.pop()
