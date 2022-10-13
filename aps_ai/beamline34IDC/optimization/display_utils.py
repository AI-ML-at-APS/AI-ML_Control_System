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

from typing import List, NoReturn

# This is just for a live plotting utility --------------------------------
import IPython
import matplotlib.pyplot as plt
import numpy as np

from aps_ai.beamline34IDC.optimization import configs
from aps_ai.beamline34IDC.optimization.common import OptimizationCommon

# Check if we are in a ipython/colab environment
try:
    class_name = IPython.get_ipython().__class__.__name__
    if "Terminal" in class_name:
        IS_NOTEBOOK = False
    else:
        IS_NOTEBOOK = True
except NameError:
    IS_NOTEBOOK = False

if IS_NOTEBOOK:
    from IPython import display


# ---------------------------------------------------------------------------

class EarlyStoppingCallback:
    """I was originally writing this to use with the scipy optimizer, but I am not using it right now."""

    def __init__(self, optimizer: OptimizationCommon, xtols: List[float] = None, ftol: float = None) -> NoReturn:
        self.optimizer = optimizer
        if xtols is None:
            self.xtols = np.array([configs.DEFAULT_MOTOR_TOLERANCES[mt] for mt in optimizer.motor_types])
        elif np.ndim(xtols) == 0:
            self.xtols = np.full(xtols, len(optimizer.motor_types))
        else:
            self.xtols = xtols

        if ftol is None:
            self.ftol = configs.DEFAULT_LOSS_TOLERANCES[optimizer.loss_parameter]

    def call(self, *args, **kwargs) -> NoReturn:
        x_prev, x_this = self.optimizer._opt_trials_motor_positions[-2:]
        loss_prev, loss_this = self.optimizer._opt_trials_losses[-2:]
        if np.all(np.abs(x_prev - x_this) < self.xtols) or np.abs(): pass


class LivePlotCallback:
    def __init__(self, optimizer: OptimizationCommon, **fig_kwargs) -> NoReturn:
        if not IS_NOTEBOOK:
            raise Exception("Cannot use live plot callback outside ipython environment for now.")
        self.optimizer = optimizer
        self._fig_initialized = False
        self.fig_kwargs = fig_kwargs

    def _initialize_fig(self) -> NoReturn:
        self.fig, self.ax = plt.subplots(1, 1, **self.fig_kwargs)
        self.hdisplay = display.display("", display_id=True)
        self.ax.set_xlabel("Calls")
        self.ax.set_ylabel("Loss")
        self.ax.set_yscale("log")
        self._fig_initialized = True

    def call(self, *args, **kwargs) -> NoReturn:

        if not self._fig_initialized:
            self._initialize_fig()
        x_points = np.arange(self.optimizer._opt_fn_call_counter)

        colors = x_points / x_points.size
        self.ax.scatter(x_points, self.optimizer._opt_trials_losses, marker='o', c=colors, cmap='jet')
        self.ax.autoscale_view()
        self.hdisplay.update(self.fig)

    def close(self) -> NoReturn:
        if self._fig_initialized:
            plt.close(self.fig)
