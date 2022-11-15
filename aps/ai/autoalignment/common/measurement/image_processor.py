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
from aps.common.initializer import IniMode, register_ini_instance, get_registered_ini_instance
from aps.common.measurment.beamline.image_processor import ImageProcessor as ImageProcessorCommon

APPLICATION_NAME = "IMAGE-PROCESSOR"

register_ini_instance(IniMode.LOCAL_FILE,
                      ini_file_name="image_processor.ini",
                      application_name=APPLICATION_NAME,
                      verbose=False)
ini_file = get_registered_ini_instance(APPLICATION_NAME)

ENERGY                = ini_file.get_float_from_ini(section="Execution", key="Energy",                default=20000.0)
SOURCE_DISTANCE_V     = ini_file.get_float_from_ini(section="Execution", key="Source-Distance-V",     default=1.5)
SOURCE_DISTANCE_H     = ini_file.get_float_from_ini(section="Execution", key="Source-Distance-H",     default=1.5)
IMAGE_TRANSFER_MATRIX = ini_file.get_list_from_ini( section="Execution", key="Image-Transfer-Matrix", default=[0, 1, 0], type=int)

ini_file.set_value_at_ini(section="Execution",   key="Energy",                value=ENERGY)
ini_file.set_value_at_ini(section="Execution",   key="Source-Distance-V",     value=SOURCE_DISTANCE_V)
ini_file.set_value_at_ini(section="Execution",   key="Source-Distance-H",     value=SOURCE_DISTANCE_H)
ini_file.set_list_at_ini( section="Execution",   key="Image-Transfer-Matrix", values_list=IMAGE_TRANSFER_MATRIX)

ini_file.push()

class ImageProcessor(ImageProcessorCommon):
    def __init__(self, data_collection_directory):
        super(ImageProcessor, self).__init__(data_collection_directory=data_collection_directory,
                                             energy=ENERGY,
                                             source_distance=[SOURCE_DISTANCE_H, SOURCE_DISTANCE_V],
                                             image_transfer_matrix=IMAGE_TRANSFER_MATRIX)


    def generate_simulated_mask(self, image_index_for_mask=1, verbose=False):
        image_transfer_matrix = super(ImageProcessor, self).generate_simulated_mask(image_index_for_mask, verbose)

        ini_file = get_registered_ini_instance(APPLICATION_NAME)
        ini_file.set_list_at_ini(section="Execution", key="Image-Transfer-Matrix", values_list=image_transfer_matrix)
        ini_file.push()
