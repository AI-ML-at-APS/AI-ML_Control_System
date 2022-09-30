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
import scipy
from aps_ai.common.util.shadow.common import EmptyBeamException
from aps_ai.beamline34IDC.optimization import common, movers, configs
from typing import List, Tuple, Callable, NoReturn


class ScipyOptimizer(common.OptimizationCommon):
    opt_platform = 'scipy'

    def set_optimizer_options(self, maxiter: int = None, maxfev: int = None,
                              xtol: float = None, ftol: float = None,
                              **extra_options) -> NoReturn:

        self._opt_params = extra_options

        if maxiter is None:
            maxiter = 50 * len(self.motor_types)
        if maxfev is None:
            maxfev = 50 * len(self.motor_types)
        if xtol is None:
            xtol = np.min([configs.DEFAULT_MOTOR_TOLERANCES[mt] for mt in self.motor_types])
        if ftol is None:
            ftol = self._loss_min_value
        if 'method' not in extra_options:
            self._opt_params['method'] = 'Nelder-Mead'
        if 'adaptive' not in extra_options:
            adaptive = True
        self._opt_params.update({'maxiter': maxiter, 'maxfev': maxfev})
        if self._opt_params['method'] == 'Nelder-Mead':
            self._opt_params.update({'xatol': xtol, 'fatol': ftol, 'adaptive': adaptive})
        else:
            self._opt_params.update({'xtol': xtol, 'ftol': ftol})

    def _get_optimizer_kwargs_options(self):
        # Default options that should be passed as kwargs and not within 'options'
        minimize_default_kwargs = ['args', 'jac', 'hess', 'hessp', 'method',
                                   'bounds', 'constraints', 'tol', 'callback']
        kwargs = {}
        options = {}
        for param, val in self._opt_params.items():
            if param in minimize_default_kwargs:
                kwargs[param] = val
            else:
                options[param] = val
        return kwargs, options

    def _optimize(self, initial_guess: List[float] = None,
                  guess_range: List[float] = None,
                  trial_count: int = 0,
                  verbose: bool = False) -> Tuple[object, List[float], bool]:

        if initial_guess is None or trial_count > 0:
            initial_guess = [np.random.uniform(m1, m2) for (m1, m2) in guess_range]

        print('initial guess is', initial_guess)

        lossfn_obj_this = self.TrialInstanceLossFunction(self, verbose=verbose)
        guess_loss = lossfn_obj_this.loss(initial_guess, verbose=False)

        while guess_loss == self._out_of_bounds_loss:
            print("Initial guess", initial_guess, "produces beam out of bounds. Trying another guess.")
            initial_guess = [np.random.uniform(m1, m2) for (m1, m2) in guess_range]
            print('initial guess is', initial_guess)
            guess_loss = lossfn_obj_this.loss(initial_guess, verbose=False)

        kwargs, options = self._get_optimizer_kwargs_options()
        opt_result = scipy.optimize.minimize(lossfn_obj_this.loss, initial_guess, **kwargs,
                                             options=options)
        loss = opt_result.fun
        sol = opt_result.x
        status = opt_result.status

        self.guesses_all.append(initial_guess)
        self.results_all.append(opt_result)

        if loss < self._loss_min_value:
            return opt_result, sol, True
        return opt_result, sol, False

    def trials(self, n_guesses: int = 5,
               initial_guess: float = None,
               verbose: bool = False,
               guess_range: List[float] = None,
               accept_all_solutions: bool = False) -> Tuple[List[object], List[float], List[float], bool]:
        """Supply zeros to the initial guess to start from the initial posiiton."""

        if guess_range is None:
            guess_range = [np.array(configs.DEFAULT_MOVEMENT_RANGES[mt]) / 2 for mt in self.motor_types]
        elif np.ndim(guess_range) == 1 and len(guess_range) == 2:
            guess_range = [guess_range for mt in self.motor_types]
        elif np.ndim(guess_range) != 2 or len(guess_range) != len(self.motor_types):
            raise ValueError("Invalid range supplied for guesses.")

        if initial_guess is not None:
            initial_guess = np.atleast_1d(initial_guess)
            if len(initial_guess) != len(self.motor_types): raise ValueError("Invalid initial guess supplied.")

        self._check_initial_loss(verbose=verbose)

        for n_trial in range(n_guesses):
            result, solution, success_status = self._optimize(guess_range=guess_range, verbose=verbose)

            if accept_all_solutions or success_status:
                return self.results_all, self.guesses_all, solution, True

            if n_trial < n_guesses:
                self.focusing_system = movers.move_motors(self.focusing_system, self.motor_types,
                                                          self.initial_motor_positions, movement='absolute')

        return self.results_all, self.guesses_all, solution, False
