#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2021, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2021. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# ----------------------------------------------------------------------- #
import numpy as np
import skopt
from beamline34IDC.util.shadow.common import EmptyBeamException
from beamline34IDC.optimization import common, movers, configs
from typing import List, Tuple, Callable, NoReturn


class SkoptGaussianProcessOptimizer(common.OptimizationCommon):
    """Changes the default optimization from scipy-based to Nelder-Mead method to
    using Gaussian Processes from skopt. Caurrently only supporting the GP optimization from skopt"""
    opt_platform = 'skopt'

    def set_optimizer_options(self, bounds=None, n_calls: int = None,  **extra_options) -> NoReturn:

        if n_calls is None:
            n_calls = 50 * len(self.motor_types)

        if bounds is None:
            bounds = []
            for mot in self.motor_types:
                bounds.append(configs.DEFAULT_MOVEMENT_RANGES[mot])
        if np.ndim(bounds) == 1:
            bounds = np.array([bounds])
        if len(bounds) != len(self.motor_types):
            raise ValueError
        self._opt_params = extra_options
        self._opt_params.update({'dimensions': bounds, 'n_calls':n_calls})

    def _optimize(self, verbose: bool = False) -> Tuple[object, List[float], bool]:

        lossfn_obj_this = self.TrialInstanceLossFunction(self, verbose=verbose)
        opt_result = skopt.gp_minimize(lossfn_obj_this.loss, **self._opt_params)
        loss = opt_result.fun
        sol = opt_result.x
        print("Loss is", loss, "for x", sol, "and min acceptable value is", self._loss_min_value)
        if loss < self._loss_min_value:
            print("Solution is acceptable.")
            return opt_result, sol, True
        print("Solution is not acceptable.")

        self.results_all.append(opt_result)
        #print('x_iters are', len(opt_result.x_iters), opt_result.x_iters)
        #print('func_vals are', len(opt_result.func_vals), opt_result.func_vals)

        # if 'x0' in self._opt_params:
        #     print('x0 is', self._opt_params['x0'])
        #     self._opt_params['x0'] = self._opt_params['x0'] + opt_result.x_iters
        #     self._opt_params['y0'] = self._opt_params['y0'] + opt_result.func_vals
        # else:
        #     self._opt_params['x0'] = opt_result.x_iters
        #     self._opt_params['y0'] = opt_result.func_vals
        # print('New x0 is', len(self._opt_params['x0']), self._opt_params['x0'])
        # print('New y0 is', len(self._opt_params['y0']), self._opt_params['y0'])
        return opt_result, sol, False

    def trials(self, n_guesses = 1, verbose: bool = False, accept_all_solutions: bool = False) -> Tuple[List[object], List[float], List[float], bool]:

        if n_guesses != 1:
            print('Warning: Since the skopt optimization samples random points anyway, there is little ' + \
                          'advantage to running multiple trials. Moreover, there is a chance the optimizer will ' + \
                          'just resample the same points again---avoiding this gets buggy and complicated ' + \
                          'and involves tinkering with the x0 and y0 paramters of gp_minimize. A simpler solution ' +\
                          'is to just increase the n_calls parameter in the optimizer options.')

        self._check_initial_loss(verbose=verbose)
        result, solution, success_status = self._optimize(verbose=verbose)

        if accept_all_solutions or success_status:
                return self.results_all, self.guesses_all, solution, True

        return self.results_all, self.guesses_all, solution, False


class SkoptDiscreteGPOptimizer(SkoptGaussianProcessOptimizer):
    class TrialInstanceLossFunction(common.OptimizationCommon.TrialInstanceLossFunction):
        def loss(self, x_absolute_this: List[float], verbose: bool = None) -> float:
            if np.ndim(x_absolute_this) > 0:
                x_absolute_this = np.array(x_absolute_this)
            x_relative_this = x_absolute_this - self.x_absolute_prev
            self.x_absolute_prev = x_absolute_this

            x_relative_this_float = self.opt_common.transform_to_float(self.opt_common.motor_types, x_relative_this)
            self.current_loss = self.opt_common.loss_function(x_relative_this_float, verbose=False)
            verbose = verbose if verbose is not None else self.verbose
            if verbose:
                print("motors", self.opt_common.motor_types,
                      "trans", x_absolute_this, "current loss", self.current_loss)
            return self.current_loss

    @staticmethod
    def transform_to_integer(motor_types: List[str], motor_values: List[float]):
        int_values = []
        for mt, mval in zip(motor_types, motor_values):
            res_this = configs.DEFAULT_MOTOR_RESOLUTIONS[mt]
            int_values.append((np.squeeze(mval) // res_this).astype('int'))
        return int_values

    @staticmethod
    def transform_to_float(motor_types: List[str], integer_values: List[int]):
        float_values = []
        for mt, mval in zip(motor_types, integer_values):
            res_this = configs.DEFAULT_MOTOR_RESOLUTIONS[mt]
            if np.ndim(mval) > 0:
                mval = np.array(mval)
            float_values.append(np.squeeze(mval) * res_this)
        return float_values

    def set_optimizer_options(self, bounds=None, n_calls: int = None,  **extra_options) -> NoReturn:

        if n_calls is None:
            n_calls = 50 * len(self.motor_types)

        if bounds is None:
            bounds = []
            for mot in self.motor_types:
                bounds.append(configs.DEFAULT_MOVEMENT_RANGES[mot])

        if np.ndim(bounds) == 1:
            bounds = np.array([bounds])
        if len(bounds) != len(self.motor_types):
            raise ValueError
        bounds_int = self.transform_to_integer(self.motor_types, bounds)
        print(bounds_int)
        self._opt_params = extra_options
        self._opt_params.update({'dimensions': bounds_int, 'n_calls':n_calls})

    def trials(self, n_guesses = 1, verbose: bool = False, accept_all_solutions: bool = False) -> Tuple[List[object], List[float], List[float], bool]:

        if n_guesses != 1:
            print('Warning: Since the skopt optimization samples random points anyway, there is little ' + \
                          'advantage to running multiple trials. Moreover, there is a chance the optimizer will ' + \
                          'just resample the same points again---avoiding this gets buggy and complicated ' + \
                          'and involves tinkering with the x0 and y0 paramters of gp_minimize. A simpler solution ' +\
                          'is to just increase the n_calls parameter in the optimizer options.')

        self._check_initial_loss(verbose=verbose)
        result, solution, success_status = self._optimize(verbose=verbose)

        if accept_all_solutions or success_status:
                return self.results_all, self.guesses_all, solution, True

        return self.results_all, self.guesses_all, solution, False