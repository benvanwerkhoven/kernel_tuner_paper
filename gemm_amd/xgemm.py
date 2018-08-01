#!/usr/bin/env python


from collections import OrderedDict
import os

import logging

import numpy as np
import kernel_tuner



def tune(do_strategy, do_method):

    path = os.path.dirname(os.path.realpath(__file__)) + "/"

    #n = np.int32(32)
    #m = np.int32(16)
    #k = np.int32(32)
    n = np.int32(2048)
    m = np.int32(2048)
    k = np.int32(2048)

    #// Matrices are accessed as follows:
    #// A: [k*M + m], with 'k' ranging from 0:K and 'm' from 0:M (m,k,m)
    #// B: [k*N + n], with 'k' ranging from 0:K and 'n' from 0:N (n,k,n)
    #// C: [n*M + m], with 'n' ranging from 0:N and 'm' from 0:M (m,n,m)

    A = np.array(np.random.randn(m, k), order='F').astype(np.float32)
    B = np.array(np.random.randn(k, n), order='F').astype(np.float32)
    #C = np.array(np.random.randn(m, n), order='F').astype(np.float32)
    C = np.zeros((m, n), order='F').astype(np.float32)

    #A = np.array(list(range(m)) * k).reshape(m, k, order='F').astype(np.float32)
    #B = np.array(list(range(k)) * n).reshape(k, n, order='F').astype(np.float32)
    #B = np.ones_like(B)
    #C = np.zeros_like(C).astype(np.float32)

    alpha, beta = np.random.randn(2).astype(np.float32)
    alpha, beta = np.array([1.0, 1.0]).astype(np.float32)

    kernel_string = ""
    files = ["common.opencl", "xgemm_part1.opencl", "xgemm_part2.opencl", "xgemm_part3.opencl"]
    for f in files:
        with open(path + f, "r") as fp:
            kernel_string += fp.read()

    args = [m, n, k, alpha, beta, A, B, C]

    tune_params = OrderedDict()

    tune_params["MWG"] = [16, 32, 64]
    tune_params["NWG"] = [16, 32, 64]
    tune_params["KWG"] = [32]
    tune_params["MDIMC"] = [8, 16, 32]
    tune_params["NDIMC"] = [8, 16, 32]
    tune_params["MDIMA"] = [8, 16, 32]
    tune_params["NDIMB"] = [8, 16, 32]
    tune_params["KWI"] = [2]
    tune_params["VWM"] = [1, 2, 4]
    tune_params["VWN"] = [1, 2, 4]
    tune_params["STRM"] = [0]
    tune_params["STRN"] = [0]
    tune_params["SA"] = [0, 1]
    tune_params["SB"] = [0, 1]
    tune_params["PRECISION"] = [32]

    problem_size = (m, n)

    grid_div_x = ["MWG"]
    grid_div_y = ["NWG"]
    block_size_names = ["MDIMC", "NDIMC", "block_size_z"]

    restrict = []
    restrict += ["KWG % KWI == 0"]
    restrict += ["MWG % (MDIMC * VWM) == 0"]
    restrict += ["NWG % (NDIMC * VWN) == 0"]
    restrict += ["MWG % (MDIMA * VWM) == 0"]
    restrict += ["NWG % (NDIMB * VWN) == 0"]
    restrict += ["KWG % ((MDIMC * NDIMC)/MDIMA) == 0"]
    restrict += ["KWG % ((MDIMC * NDIMC)/NDIMB) == 0"]

    C_ref = (np.dot(alpha * A, B.T) + beta * C).astype(np.float32)


    answer = [None for _ in args]
    answer[-1] = C_ref


    results, env = kernel_tuner.tune_kernel("Xgemm", kernel_string, problem_size, args, tune_params, block_size_names=block_size_names,
                             lang="OpenCL", platform=1, device=0, restrictions=restrict, verbose=True, compiler_options=["-I"+path],
                             grid_div_x=grid_div_x, grid_div_y=grid_div_y, answer=answer, atol=1e-2,
                             strategy=do_strategy, method=do_method)

    return results, env


if __name__ == "__main__":
    tune(None, None)
