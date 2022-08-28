#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2022, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2022. UChicago Argonne, LLC. This software was produced       #
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
import os.path

import numpy
import Shadow
from Shadow import ShadowTools as ST


from orangecontrib.shadow.util.shadow_objects import ShadowOpticalElement
from orangecontrib.shadow.widgets.special_elements.bl import hybrid_control

from beamline34IDC.util.shadow.common import HybridFailureException, rotate_axis_system, get_hybrid_input_parameters
from beamline34IDC.facade.focusing_optics_interface import Movement, MotorResolution, AngularUnits, DistanceUnits

from beamline34IDC.simulation.shadow.focusing_optics.abstract_focusing_optics import FocusingOpticsCommon
from beamline34IDC.simulation.shadow.focusing_optics.bender.calibrated_bender import CalibratedBenderManager, HKBMockWidget, VKBMockWidget
from beamline34IDC.simulation.facade.focusing_optics_interface import get_default_input_features

from oasys.widgets.abstract.benders.double_rod_bendable_ellispoid_mirror import ideal_height_profile, BenderDataToPlot
from orangecontrib.shadow_advanced_tools.widgets.optical_elements.bl.double_rod_bendable_ellispoid_mirror_bl import apply_bender_surface

class CalibratedBendableFocusingOptics(FocusingOpticsCommon):
    def __init__(self):
        super(FocusingOpticsCommon, self).__init__()

    def initialize(self,
                   input_photon_beam,
                   input_features=get_default_input_features(),
                   **kwargs):

        super().initialize(input_photon_beam, input_features, **kwargs)

        self.__vkb_bender_manager = CalibratedBenderManager(kb_raytracing = VKBMockWidget(shadow_oe=self._vkb, verbose=True, label="Raytracing"),
                                                            kb_upstream   = VKBMockWidget(shadow_oe=self._vkb.duplicate(), verbose=True, label="Upstream"),
                                                            kb_downstream = VKBMockWidget(shadow_oe=self._vkb.duplicate(), verbose=True, label="Downstream"))
        self.__vkb_bender_manager.load_calibration("V-KB")
        self.__vkb_bender_manager.set_positions(input_features.get_parameter("vkb_motor_1_bender_position"),
                                                input_features.get_parameter("vkb_motor_2_bender_position"))
        self.__vkb_bender_manager.remove_bender_files()

        self.__hkb_bender_manager = CalibratedBenderManager(kb_raytracing = HKBMockWidget(shadow_oe=self._hkb, verbose=True, label="Raytracing"),
                                                            kb_upstream   = HKBMockWidget(shadow_oe=self._hkb.duplicate(), verbose=True, label="Upstream"),
                                                            kb_downstream = HKBMockWidget(shadow_oe=self._hkb.duplicate(), verbose=True, label="Downstream"))

        self.__hkb_bender_manager.load_calibration("H-KB")
        self.__hkb_bender_manager.set_positions(input_features.get_parameter("hkb_motor_1_bender_position"),
                                                input_features.get_parameter("hkb_motor_2_bender_position"))
        self.__hkb_bender_manager.remove_bender_files()

    def _initialize_kb(self, input_features, reflectivity_file, vkb_error_profile_file, hkb_error_profile_file):
        # V-KB --------------------
        vkb_motor_3_pitch_angle             = input_features.get_parameter("vkb_motor_3_pitch_angle")
        vkb_pitch_angle_shadow              = 90 - numpy.degrees(vkb_motor_3_pitch_angle)
        vkb_motor_3_delta_pitch_angle       = input_features.get_parameter("vkb_motor_3_delta_pitch_angle")
        vkb_pitch_angle_displacement_shadow = numpy.degrees(vkb_motor_3_delta_pitch_angle)
        vkb_motor_4_translation             = input_features.get_parameter("vkb_motor_4_translation")

        vkb = Shadow.OE()
        vkb.ALPHA = 180.0
        vkb.DUMMY = 0.1
        vkb.FCYL = 1
        vkb.FHIT_C = 1
        vkb.FILE_REFL = reflectivity_file.encode()
        vkb.FILE_RIP = vkb_error_profile_file.encode()
        vkb.FMIRR = 2
        vkb.FWRITE = 1
        vkb.F_DEFAULT = 0
        vkb.F_G_S = 2
        vkb.F_REFLEC = 1
        vkb.F_RIPPLE = 1
        vkb.RLEN1 = 50.0
        vkb.RLEN2 = 50.0
        vkb.RWIDX1 = 20.95
        vkb.RWIDX2 = 20.95
        vkb.SIMAG = -999
        vkb.SSOUR = 50667.983
        vkb.THETA = vkb_pitch_angle_shadow
        vkb.T_IMAGE = 108.0
        vkb.T_INCIDENCE = vkb_pitch_angle_shadow
        vkb.T_REFLECTION = vkb_pitch_angle_shadow
        vkb.T_SOURCE = 150.0

        # DISPLACEMENTS
        vkb.F_MOVE = 1
        vkb.OFFY = vkb_motor_4_translation * numpy.sin(vkb_motor_3_pitch_angle + vkb_motor_3_delta_pitch_angle)
        vkb.OFFZ = vkb_motor_4_translation * numpy.cos(vkb_motor_3_pitch_angle + vkb_motor_3_delta_pitch_angle)
        vkb.X_ROT = vkb_pitch_angle_displacement_shadow

        # H-KB --------------------

        hkb_motor_3_pitch_angle = input_features.get_parameter("hkb_motor_3_pitch_angle")
        hkb_pitch_angle_shadow = 90 - numpy.degrees(hkb_motor_3_pitch_angle)
        hkb_motor_3_delta_pitch_angle = input_features.get_parameter("hkb_motor_3_delta_pitch_angle")
        hkb_pitch_angle_displacement_shadow = numpy.degrees(hkb_motor_3_delta_pitch_angle)
        hkb_motor_4_translation = input_features.get_parameter("hkb_motor_4_translation")

        hkb = Shadow.OE()
        hkb.ALPHA = 90.0
        hkb.DUMMY = 0.1
        hkb.FCYL = 1
        hkb.FHIT_C = 1
        hkb.FILE_REFL = reflectivity_file.encode()
        hkb.FILE_RIP = hkb_error_profile_file.encode()
        hkb.FMIRR = 2
        hkb.FWRITE = 1
        hkb.F_DEFAULT = 0
        hkb.F_G_S = 2
        hkb.F_REFLEC = 1
        hkb.F_RIPPLE = 1
        hkb.RLEN1 = 50.0  # dim plus
        hkb.RLEN2 = 50.0  # dim minus
        hkb.RWIDX1 = 24.75
        hkb.RWIDX2 = 24.75
        hkb.SIMAG = -999
        hkb.SSOUR = 50775.983
        hkb.THETA = hkb_pitch_angle_shadow
        hkb.T_IMAGE = 123.0
        hkb.T_INCIDENCE = hkb_pitch_angle_shadow
        hkb.T_REFLECTION = hkb_pitch_angle_shadow
        hkb.T_SOURCE = 0.0

        # DISPLACEMENT
        hkb.F_MOVE = 1
        hkb.X_ROT = hkb_pitch_angle_displacement_shadow
        hkb.OFFY = hkb_motor_4_translation * numpy.sin(hkb_motor_3_pitch_angle + hkb_motor_3_delta_pitch_angle)
        hkb.OFFZ = hkb_motor_4_translation * numpy.cos(hkb_motor_3_pitch_angle + hkb_motor_3_delta_pitch_angle)

        self._vkb = ShadowOpticalElement(vkb)
        self._hkb = ShadowOpticalElement(hkb)

    # ---- H-KB ---------------------------------------------------------

    def move_vkb_motor_1_bender(self, pos_upstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_motor_1_2_bender(self.__vkb_bender_manager, pos_upstream, None, movement, units,
                                     round_digit=MotorResolution.getInstance().get_vkb_motor_1_2_bender_resolution(units=DistanceUnits.MICRON)[1])

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_motor_1_bender(self, units=DistanceUnits.MICRON):
        return self.__get_motor_1_2_bender(self.__vkb_bender_manager, units)[0]

    def move_vkb_motor_2_bender(self, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_motor_1_2_bender(self.__vkb_bender_manager, None, pos_downstream, movement, units,
                                     round_digit=MotorResolution.getInstance().get_vkb_motor_1_2_bender_resolution(units=DistanceUnits.MICRON)[1])

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_motor_2_bender(self, units=DistanceUnits.MICRON):
        return self.__get_motor_1_2_bender(self.__vkb_bender_manager, units)[1]

    def get_vkb_q_distance(self):
        return self.__vkb_bender_manager.get_q_distances()

    def move_vkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS):
        self._move_motor_3_pitch(self._vkb, angle, movement, units,
                                 round_digit=MotorResolution.getInstance().get_vkb_motor_3_pitch_resolution(units=AngularUnits.DEGREES)[1], invert=True)

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS):
        # motor 3/4 are identical for the two sides
        return self._get_motor_3_pitch(self._vkb, units, invert=True)

    def move_vkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self._move_motor_4_transation(self._vkb, translation, movement, units,
                                      round_digit=MotorResolution.getInstance().get_vkb_motor_4_translation_resolution(units=DistanceUnits.MILLIMETERS)[1], invert=True)

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_motor_4_translation(self, units=DistanceUnits.MICRON):
        # motor 3/4 are identical for the two sides
        return self._get_motor_4_translation(self._vkb, units, invert=True)

    # ---- H-KB ---------------------------------------------------------

    def move_hkb_motor_1_bender(self, pos_upstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_motor_1_2_bender(self.__hkb_bender_manager, pos_upstream, None, movement, units,
                                     round_digit=MotorResolution.getInstance().get_hkb_motor_1_2_bender_resolution(units=DistanceUnits.MICRON)[1])

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_motor_1_bender(self, units=DistanceUnits.MICRON):
        return self.__get_motor_1_2_bender(self.__hkb_bender_manager, units)[0]

    def move_hkb_motor_2_bender(self, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_motor_1_2_bender(self.__hkb_bender_manager, None, pos_downstream, movement, units,
                                     round_digit=MotorResolution.getInstance().get_hkb_motor_1_2_bender_resolution(units=DistanceUnits.MICRON)[1])

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_motor_2_bender(self, units=DistanceUnits.MICRON):
        return self.__get_motor_1_2_bender(self.__hkb_bender_manager, units)[1]

    def get_hkb_q_distance(self):
        return self.__hkb_bender_manager.get_q_distances()

    def move_hkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS):
        self._move_motor_3_pitch(self._hkb, angle, movement, units,
                                 round_digit=MotorResolution.getInstance().get_hkb_motor_3_pitch_resolution(units=AngularUnits.DEGREES)[1])

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS):
        # motor 3/4 are identical for the two sides
        return self._get_motor_3_pitch(self._hkb, units)

    def move_hkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self._move_motor_4_transation(self._hkb, translation, movement, units,
                                      round_digit=MotorResolution.getInstance().get_hkb_motor_4_translation_resolution(units=DistanceUnits.MILLIMETERS)[1])

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_motor_4_translation(self, units=DistanceUnits.MICRON):
        # motor 3/4 are identical for the two sides
        return self._get_motor_4_translation(self._hkb, units)

    # IMPLEMENTATION OF PROTECTED METHODS FROM SUPERCLASS

    def _trace_vkb(self, random_seed, remove_lost_rays, verbose):
        output_beam = self.__trace_kb(bender_manager=self.__vkb_bender_manager,
                                      input_beam=self._slits_beam,
                                      widget_class_name="DoubleRodBenderEllypticalMirror",
                                      oe_name="V-KB",
                                      near_field_calculation=False, # NOTE: Near field not possible for vkb (beam is untraceable)
                                      increment=200,
                                      random_seed=random_seed,
                                      remove_lost_rays=remove_lost_rays,
                                      verbose=verbose)

        return output_beam

    def _trace_hkb(self, near_field_calculation, random_seed, remove_lost_rays, verbose):
        output_beam = \
            self.__trace_kb(bender_manager=self.__hkb_bender_manager,
                            input_beam=self._vkb_beam,
                            widget_class_name="DoubleRodBenderEllypticalMirror",
                            oe_name="H-KB",
                            near_field_calculation=near_field_calculation,
                            increment=300,
                            random_seed=random_seed,
                            remove_lost_rays=remove_lost_rays,
                            verbose=verbose)

        return rotate_axis_system(output_beam, rotation_angle=270.0)

    # PRIVATE METHODS

    def __trace_kb(self, bender_manager, input_beam, widget_class_name, oe_name, near_field_calculation, increment, random_seed, remove_lost_rays, verbose):
        upstream_widget   = bender_manager._kb_upstream
        downstream_widget = bender_manager._kb_downstream
        raytracing_widget = bender_manager._kb_raytracing

        def calculate_bender(widget):
            widget.R0 = widget.R0_out  # use last fit result
            widget.shadow_oe._oe.FILE_RIP = bytes(widget.ms_defect_file_name, 'utf-8')  # restore original error profile

            return apply_bender_surface(widget=widget, shadow_oe=widget.shadow_oe)

        q_upstream, q_downstream = bender_manager.get_q_distances()

        if (q_upstream != bender_manager.q_upstream_previous) or (q_downstream != bender_manager.q_downstream_previous) or \
                (not os.path.exists(upstream_widget.output_file_name_full)) or (not os.path.exists(downstream_widget.output_file_name_full)):
            upstream_bender_data   = calculate_bender(upstream_widget)
            downstream_bender_data = calculate_bender(downstream_widget)

            upstream_mirror_profile   = upstream_bender_data.z_bender_correction + upstream_bender_data.ideal_profile
            downstream_mirror_profile = downstream_bender_data.z_bender_correction + downstream_bender_data.ideal_profile

            dim_y = len(upstream_bender_data.y)

            raytracing_mirror_profile = numpy.zeros(upstream_mirror_profile.shape)

            raytracing_mirror_profile[:, 0:dim_y] = upstream_mirror_profile[:, 0:dim_y]
            raytracing_mirror_profile[:, dim_y:]  = downstream_mirror_profile[:, dim_y:]

            ideal_profile = ideal_height_profile(y=upstream_bender_data.y,
                                                 p=raytracing_widget.object_side_focal_distance,
                                                 q=raytracing_widget.image_side_focal_distance,
                                                 grazing_angle=numpy.radians(90 - raytracing_widget.incidence_angle_respect_to_normal))
            ideal_profile -= numpy.min(ideal_profile)

            raytracing_ideal_profile = numpy.zeros(upstream_mirror_profile.shape)
            for i in range(raytracing_ideal_profile.shape[0]): raytracing_ideal_profile[i, :] = numpy.copy(ideal_profile)

            z_bender_correction = raytracing_mirror_profile-raytracing_ideal_profile

            ST.write_shadow_surface(z_bender_correction.T, numpy.round(upstream_bender_data.x, 6), numpy.round(upstream_bender_data.y, 6), raytracing_widget.output_file_name_full)

            from matplotlib import cm
            from matplotlib import pyplot as plt

            figure = plt.figure(figsize=(10, 7))
            figure.patch.set_facecolor('white')

            axis = figure.add_subplot(111, projection='3d')
            axis.set_zlabel("Z [nm]")
            axis.set_xlabel("X [mm]")
            axis.set_ylabel("Y [mm]")

            x_to_plot, y_to_plot = numpy.meshgrid(upstream_bender_data.x, upstream_bender_data.y)
            z_to_plot = z_bender_correction.T * 1e6

            axis.plot_surface(x_to_plot, y_to_plot, z_to_plot, rstride=1, cstride=1, cmap=cm.autumn, linewidth=0.5, antialiased=True)
            plt.show()

        raytracing_widget.shadow_oe._oe.F_RIPPLE = 1
        raytracing_widget.shadow_oe._oe.F_G_S = 2
        raytracing_widget.shadow_oe._oe.FILE_RIP = bytes(raytracing_widget.output_file_name_full, 'utf-8')

        bender_manager.q_upstream_previous   = q_upstream
        bender_manager.q_downstream_previous = q_downstream

        # Redo raytracing with the bender correction as error profile
        output_beam = self._trace_oe(input_beam=input_beam,
                                     shadow_oe=raytracing_widget.shadow_oe,
                                     widget_class_name=widget_class_name,
                                     oe_name=oe_name,
                                     remove_lost_rays=remove_lost_rays)
        def run_hybrid(output_beam):
            try:
                if not near_field_calculation:
                    return hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                             diffraction_plane=2,  # Tangential
                                                                             calcType=3,  # Diffraction by Mirror Size + Errors
                                                                             verbose=verbose,
                                                                             random_seed=None if random_seed is None else (random_seed + increment))).ff_beam
                else:
                    return hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                             diffraction_plane=2,  # Tangential
                                                                             calcType=3,  # Diffraction by Mirror Size + Errors
                                                                             nf=1,
                                                                             verbose=verbose,
                                                                             random_seed=None if random_seed is None else (random_seed + increment + 1))).nf_beam
            except Exception:
                raise HybridFailureException(oe=oe_name)

        return run_hybrid(output_beam)

    @classmethod
    def __move_motor_1_2_bender(cls, bender_manager, pos_upstream, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON, round_digit=2):
        if bender_manager is None: raise ValueError("Initialize Focusing Optics System first")

        current_pos_upstream, current_pos_downstream = bender_manager.get_positions()

        def check_pos(pos, current_pos):
            if not pos is None:
                if units == DistanceUnits.MILLIMETERS:
                    return round(pos * 1e3, round_digit)
                elif units == DistanceUnits.MICRON:
                    return round(pos, round_digit)
                else:
                    raise ValueError("Distance units not recognized")
            else:
                return 0.0 if movement == Movement.RELATIVE else current_pos

        pos_upstream = check_pos(pos_upstream, current_pos_upstream)
        pos_downstream = check_pos(pos_downstream, current_pos_downstream)

        if movement == Movement.ABSOLUTE:
            bender_manager.set_positions(pos_upstream, pos_downstream)
        elif movement == Movement.RELATIVE:
            current_pos_upstream, current_pos_downstream = bender_manager.get_positions()
            bender_manager.set_positions(current_pos_upstream + pos_upstream, current_pos_downstream + pos_downstream)
        else:
            raise ValueError("Movement not recognized")

    @classmethod
    def __get_motor_1_2_bender(cls, bender_manager, units=DistanceUnits.MICRON):
        if bender_manager is None: raise ValueError("Initialize Focusing Optics System first")

        pos_upstream, pos_downstream = bender_manager.get_positions()

        if units == DistanceUnits.MILLIMETERS:
            pos_upstream *= 1e-3
            pos_downstream *= 1e-3
        elif units == DistanceUnits.MICRON:
            pass
        else:
            raise ValueError("Distance units not recognized")

        return pos_upstream, pos_downstream
