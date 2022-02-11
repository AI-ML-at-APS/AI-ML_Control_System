"""
** General notes **:
The screen is very small, so the range of motion before the beam lands outside is quite small.
We can incorporate this physical constraints as bounds in the optimization method.

Maximum allowable movements (from screen edge-to-edge):
    - Motor translations: +/- 500 microns
    - Pitch angle: +/- 0.2 radians
    - Curvature: +/- 20 mm

---------------------------
** Optimization notes ** :

Modify the optimization "loss function" to optimize for the position of the beam
(so that it hits the center of the screen), and not the peak intensity or FWHM.
The primary objective of the focusing procedure is to ensure the beam hits the center of the screen.
We expect that the crude focusing procedure already produces a reasonably tight coherent beam,
and there is not much room for play in the beam size.

Optimizing the beam size or beam intensity can be a secondary objective.
An alternating procedure optimizing the position first, then the beam size or beam intensity would be ok.

Initial work:
    - Assume that the beam curvature and Q are ok.
    - Only adjust the horizontal and vertical motors.
    - First start with one of these parameters, then try optimizing both simultaneously.
    - Use "relative movement" for the motors.

"""

import os
from beamline34IDC.simulation.facade import Implementors
from beamline34IDC.simulation.facade.focusing_optics_factory import focusing_optics_factory_method
from beamline34IDC.simulation.facade.focusing_optics_interface import Movement

from beamline34IDC.util.shadow.common import \
    plot_shadow_beam_spatial_distribution, get_shadow_beam_spatial_distribution,\
    load_shadow_beam, PreProcessorFiles, EmptyBeamException
from beamline34IDC.util import clean_up
import matplotlib.pyplot as plt
import numpy as np
import scipy
from beamline34IDC.util.redirect_shadow_output import redirected_output

#%%
DEFAULT_RANDOM_SEED = 111

#%%
def getBeam(random_seed=DEFAULT_RANDOM_SEED, remove_lost_rays=True, redirect_output=True):
    global focusing_system
    if redirect_output:
        with redirected_output():
            out_beam = focusing_system.get_photon_beam(random_seed=random_seed, remove_lost_rays=remove_lost_rays)
    else:
        out_beam = focusing_system.get_photon_beam(random_seed=random_seed, remove_lost_rays=remove_lost_rays)
    return out_beam


def getPeakIntensity(random_seed=DEFAULT_RANDOM_SEED, redirect_output=True):
    try:
        out_beam = getBeam(random_seed=random_seed, redirect_output=redirect_output)
    except EmptyBeamException:
        # Assuming that the beam is outside the screen and returning 0 as a default value.
        return 0
    _, dw = get_shadow_beam_spatial_distribution(out_beam)
    peak = dw.get_parameter('peak_intensity')
    return peak

def getCentroidDistance(random_seed=DEFAULT_RANDOM_SEED, redirect_output=True):
    global focusing_system
    try:
        out_beam = getBeam(random_seed=random_seed, redirect_output=redirect_output)
    except EmptyBeamException:
        # Assuming that the centroid is outside the screen and returning 0.5 microns as a default value.
        return 0.5
    hist, dw = get_shadow_beam_spatial_distribution(out_beam)
    h_centroid = dw.get_parameter('h_centroid')
    v_centroid = dw.get_parameter('v_centroid')
    centroid_distance = (h_centroid ** 2 + v_centroid ** 2) ** 0.5
    return centroid_distance


def moveRelative(trans):
    global focusing_system
    focusing_system.move_vkb_motor_4_translation(trans, movement=Movement.RELATIVE)


def moveAbsolute(trans):
    global focusing_system
    focusing_system.move_vkb_motor_4_translation(trans, movement=Movement.ABSOLUTE)


def lossFunction(trans, random_seed=DEFAULT_RANDOM_SEED, redirect_output=True, movement='relative'):
    global focusing_system
    if movement == 'relative':
        moveRelative(trans)
    elif movement == 'absolute':
        moveAbsolute(trans)
    else:
        raise ValueError
    centroid_distance = getCentroidDistance(random_seed=random_seed, redirect_output=redirect_output)
    return centroid_distance


def reinitialize(random_seed=DEFAULT_RANDOM_SEED, remove_lost_rays=True):
    clean_up()
    focusing_system = focusing_optics_factory_method(implementor=Implementors.SHADOW)

    focusing_system.initialize(input_photon_beam=input_beam,
                               rewrite_preprocessor_files=PreProcessorFiles.NO,
                               rewrite_height_error_profile_files=False)
    # Not redirecting the output for the initialization step, just in case there are any errors.
    output_beam = focusing_system.get_photon_beam(random_seed=random_seed, remove_lost_rays=remove_lost_rays)
    return focusing_system, output_beam

#%%
work_dir = '/Users/saugat/code/oasys/ML_Control_System/work_directory'
os.chdir(work_dir)
input_beam = load_shadow_beam("primary_optics_system_beam.dat")

#%%

#%%
# Focusing Optics System -------------------------

focusing_system, out_beam = reinitialize()

#%%
