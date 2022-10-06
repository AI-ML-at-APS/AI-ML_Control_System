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

from aps_ai.beamline34IDC.optimization import configs
from aps_ai.beamline34IDC.optimization import common as opt_common
from typing import NoReturn, List, Tuple
import gym
import numpy as np


class BareOptimization(opt_common.OptimizationCommon):
    def _optimize(self, *args, **kwargs) -> NoReturn:
        pass

    def set_optimizer_options(self, *args, **kwargs) -> NoReturn:
        pass

    def trials(self, *args, **kwargs) -> NoReturn:
        pass

class AdaptiveCameraEnv(gym.Env):
    """Where the camera center and resolution is adaptive.

    Observation space contains current camera 'hh' and 'vv' positions and the 2d histogram of counts."""
    def __init__(self, focusing_system: object,
                 motor_types: List[str],
                 loss_parameters: List[str] = ['centroid'],
                 camera_nbins: int = 256,
                 camera_xrange: List[float] = None,
                 camera_yrange: List[float] = None,
                 random_seed: int = None,
                 loss_min_value: float = 1e-6,
                 use_gaussian_fit: bool = False,
                 camera_count_scaling_factor: float = 1.0,
                 verbose: bool = False):
        super().__init__()
        self._verbose = verbose
        self.optimizer = BareOptimization(focusing_system,
                                          motor_types,
                                          random_seed=random_seed,
                                          loss_parameters=loss_parameters,
                                          camera_xrange=camera_xrange,
                                          camera_yrange=camera_yrange,
                                          camera_nbins=camera_nbins,
                                          loss_min_value=loss_min_value,
                                          use_gaussian_fit=use_gaussian_fit)

        resolutions = []
        for mt in self.optimizer.motor_types:
            res = configs.DEFAULT_MOTOR_RESOLUTIONS[mt]
            resolutions.append(res)
        self.resolutions = np.array(resolutions)

        camera_xrange_lims = [-np.inf, np.inf] if camera_xrange is None else camera_xrange
        camera_yrange_lims = [-np.inf, np.inf] if camera_yrange is None else camera_yrange

        self.camera_count_scaling_factor = camera_count_scaling_factor
        self.observation_space = gym.spaces.Dict({
            'hh': gym.spaces.Box(*camera_xrange_lims, shape=[camera_nbins], dtype='float32'),
            'vv': gym.spaces.Box(*camera_yrange_lims, shape=[camera_nbins], dtype='float32'),
            'data_2D': gym.spaces.Box(0., 255, shape=[camera_nbins, camera_nbins], dtype='int32')
        })

        # Setting an action space of 20 steps (10 positive, 10 negative) for each motor.
        self.action_space = gym.spaces.MultiDiscrete([20] * len(self.optimizer.motor_types))
        self.action_space_origs = np.array([10] * len(self.optimizer.motor_types))

        self.current_loss = self.optimizer.loss_function(0, verbose=False)
        self.current_reward = self.reward()
        self.counter = 0
        self.initialization_range_per_motor = None

    def step(self, action: object) -> Tuple:
        # this gives the loss due to the relative motion "action"
        action_this = action - self.action_space_origs

        self.current_loss = self.optimizer.loss_function(action_this * self.resolutions, verbose=False)
        self.counter += 1

        data_2D = self.optimizer.beam_state.hist.data_2D / self.camera_count_scaling_factor
        current_obs = {'hh': self.optimizer.beam_state.hist.hh.astype('float32'),
                       'vv': self.optimizer.beam_state.hist.vv.astype('float32'),
                       'data_2D': data_2D.astype('int32')}
        if current_obs['data_2D'].max() > 255:
            raise ValueError("Camera counts must be scaled to be between 0 and 255. Supply a scaling factor > 1.")

        # done = True if self.current_loss <= self.optimizer._loss_min_value else False
        done = True if self.optimizer.beam_state.hist.data_2D.sum() == 0 else False
        info = {}
        if self._verbose:
            print("Current loss is", self.current_loss, "for action", action, "at", self.counter)
        return current_obs, self.reward(), done, info

    def reward(self) -> float:
        self.current_reward = 1 - self.current_loss
        return self.current_reward

    def set_initialization_range_per_motor(self, range_per_motor: List[float]):
        self.initialization_range_per_motor = range_per_motor

    def reset(self) -> object:
        self.optimizer.reset()

        initial_motor_vals = self.optimizer.get_random_init(guess_range=self.initialization_range_per_motor,
                                                            verbose=self._verbose)
        #self.optimizer.initial_motor_positions = initial_motor_vals

        self.current_loss = self.optimizer.loss_function(initial_motor_vals, verbose=False)
        data_2D = self.optimizer.beam_state.hist.data_2D / self.camera_count_scaling_factor
        current_obs = {'hh': self.optimizer.beam_state.hist.hh.astype('float32'),
                       'vv': self.optimizer.beam_state.hist.vv.astype('float32'),
                       'data_2D': data_2D.astype('int32')}
        if current_obs['data_2D'].max() > 255:
            raise ValueError("Camera counts must be scaled to be between 0 and 255. Supply a scaling factor > 1.")
        if self._verbose:
            print("Current loss is", self.current_loss, "at counter", self.counter)
        return current_obs


class FixedCameraEnv(gym.Env):
    """Where the camera center and resolution is fixed.

    Observation space only contains the 2d histogram of counts."""
    def __init__(self, focusing_system: object,
                 motor_types: List[str],
                 loss_parameters: List[str] = ['centroid'],
                 camera_nbins: int = 256,
                 camera_xrange: List[float] = [-0.5, 0.5],
                 camera_yrange: List[float] = [-0.5, 0.5],
                 random_seed: int = None,
                 loss_min_value: float = 1e-6,
                 use_gaussian_fit: bool = False,
                 camera_count_scaling_factor: float = 1.0,
                 verbose: bool = False):
        super().__init__()
        self._verbose = verbose
        self.optimizer = BareOptimization(focusing_system,
                                          motor_types,
                                          random_seed=random_seed,
                                          loss_parameters=loss_parameters,
                                          camera_xrange=camera_xrange,
                                          camera_yrange=camera_yrange,
                                          camera_nbins=camera_nbins,
                                          loss_min_value=loss_min_value,
                                          use_gaussian_fit=use_gaussian_fit)

        resolutions = []
        for mt in self.optimizer.motor_types:
            res = configs.DEFAULT_MOTOR_RESOLUTIONS[mt]
            resolutions.append(res)

        self.camera_count_scaling_factor = camera_count_scaling_factor
        self.observation_space = gym.spaces.Dict({
            'data_2D': gym.spaces.Box(0., 255, shape=[camera_nbins, camera_nbins], dtype='int32')
        })

        # Setting an action space of 20 steps (10 positive, 10 negative) for each motor.
        self.action_space = gym.spaces.MultiDiscrete([20] * len(self.optimizer.motor_types))
        self.action_space_origs = np.array([10] * len(self.optimizer.motor_types))

        self.resolutions = np.array(resolutions)

        self.current_loss = self.optimizer.loss_function(0, verbose=False)
        self.current_reward = self.reward()
        self.counter = 0

        self.initialization_range_per_motor = None

    def step(self, action: object) -> Tuple:
        # this gives the loss due to the relative motion "action"
        action_this = action - self.action_space_origs

        self.current_loss = self.optimizer.loss_function(action_this * self.resolutions, verbose=False)
        self.counter += 1

        data_2D = self.optimizer.beam_state.hist.data_2D / self.camera_count_scaling_factor
        current_obs = {'data_2D': data_2D.astype('int32')}
        if current_obs['data_2D'].max() > 255:
            raise ValueError("Camera counts must be scaled to be between 0 and 255. Supply a scaling factor > 1.")

        # done = True if self.current_loss <= self.optimizer._loss_min_value else False
        done = True if data_2D.sum() == 0 else False
        info = {}
        if self._verbose:
            print("Current loss is", self.current_loss, "for action", action, "at", self.counter)
        return current_obs, self.reward(), done, info

    def reward(self) -> float:
        self.current_reward = 1 - self.current_loss
        return self.current_reward

    def set_initialization_range_per_motor(self, range_per_motor: List[float]):
        self.initialization_range_per_motor = range_per_motor

    def reset(self) -> object:
        self.optimizer.reset()

        initial_motor_vals = self.optimizer.get_random_init(guess_range=self.initialization_range_per_motor,
                                                            verbose=self._verbose)
        #self.optimizer.initial_motor_positions = initial_motor_vals

        self.current_loss = self.optimizer.loss_function(initial_motor_vals, verbose=False)
        data_2D = self.optimizer.beam_state.hist.data_2D / self.camera_count_scaling_factor
        current_obs = {'data_2D': data_2D.astype('int32')}
        if current_obs['data_2D'].max() > 255:
            raise ValueError("Camera counts must be scaled to be between 0 and 255. Supply a scaling factor > 1.")

        if self._verbose:
            print("Current loss is", self.current_loss, "at counter", self.counter)
        return current_obs


