import os
from beamline34IDC.simulation.facade import Implementors
from beamline34IDC.simulation.facade.focusing_optics_factory import focusing_optics_factory_method
from beamline34IDC.simulation.facade.focusing_optics_interface import Movement, AngularUnits

from beamline34IDC.util.shadow.common import \
    plot_shadow_beam_spatial_distribution, get_shadow_beam_spatial_distribution,\
    load_shadow_beam, PreProcessorFiles, EmptyBeamException
from beamline34IDC.util import clean_up
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RANDOM_SEED = 111

def reinitialize(input_beam_path):
    clean_up()

    input_beam = load_shadow_beam(input_beam_path)
    focusing_system = focusing_optics_factory_method(implementor=Implementors.SHADOW)

    focusing_system.initialize(input_photon_beam=input_beam,
                               rewrite_preprocessor_files=PreProcessorFiles.NO,
                               rewrite_height_error_profile_files=False)
    return focusing_system

#%%
print("Initial dir", os.getcwd())
os.chdir("../../work_directory")
print("Work dir", os.getcwd())
#%%
input_beam_path = "primary_optics_system_beam.dat"
focusing_system = reinitialize(input_beam_path=input_beam_path)

#%% Trying out a relative motion
initial_angle = focusing_system.get_vkb_motor_3_pitch()

delta_grazing_angle = -5
focusing_system.move_vkb_motor_3_pitch(delta_grazing_angle, movement=Movement.RELATIVE)
new_angle = focusing_system.get_vkb_motor_3_pitch()
out_beam = focusing_system.get_photon_beam()
print("Initial value", initial_angle, "move delta", delta_grazing_angle, "gives new value", new_angle)

_ = plot_shadow_beam_spatial_distribution(out_beam)
plt.show(block=True)
#%% Trying out another relative motion without reinitializing
initial_angle = focusing_system.get_vkb_motor_3_pitch()
delta_grazing_angle = 0

focusing_system.move_vkb_motor_3_pitch(delta_grazing_angle, movement=Movement.RELATIVE)
new_angle = focusing_system.get_vkb_motor_3_pitch()
out_beam = focusing_system.get_photon_beam()
print("Initial value", initial_angle, "move delta", delta_grazing_angle, "gives new value", new_angle)

_ = plot_shadow_beam_spatial_distribution(out_beam)
plt.show(block=True)

## Result: seems like every move_vkb_motor_3_pitch call is reinitializing the focusing system?
#%%