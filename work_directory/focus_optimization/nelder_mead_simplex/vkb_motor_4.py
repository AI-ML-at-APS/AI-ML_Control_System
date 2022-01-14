#%%
import os
import sys
sys.path.append('/Users/saugat/code/oasys/ML_Control_System/work_directory/focus_optimization/nelder_mead_simplex')

from beamline34IDC.simulation.focusing_optics_system import FocusingOpticsSystem, Movement
from beamline34IDC.util.common import PreProcessorFiles, plot_shadow_beam_spatial_distribution, \
    load_shadow_beam, get_shadow_beam_spatial_distribution
from beamline34IDC.util import clean_up
import matplotlib.pyplot as plt

from gaussian_fit import calculate_gaussian_fit
import scipy

#%%
def getPeakIntensity(focusing_system):
    out_beam = focusing_system.get_beam()
    _, dw = get_shadow_beam_spatial_distribution(out_beam)
    peak = dw.get_parameter('peak_intensity')
    return out_beam, peak

def loss_function(focusing_system, trans, move_type = Movement.RELATIVE):
    focusing_system.move_vkb_motor_4_translation(trans, movement=move_type)
    _, peak = getPeakIntensity(focusing_system)
    return peak

def getAmplitudeFromGaussianFit(focusing_system):
    out_beam = focusing_system.get_beam()
    hist, dw = get_shadow_beam_spatial_distribution(out_beam)
    pass

#%%
work_dir = '/Users/saugat/code/oasys/ML_Control_System/work_directory'
os.chdir(work_dir)
input_beam = load_shadow_beam("primary_optics_system_beam.dat")

#%%
# Focusing Optics System -------------------------

focusing_system = FocusingOpticsSystem()

focusing_system.initialize(input_beam=input_beam,
                           rewrite_preprocessor_files=PreProcessorFiles.NO,
                           rewrite_height_error_profile_files=False)
#%%
out_beam = focusing_system.get_beam()
plot_shadow_beam_spatial_distribution(out_beam)
plt.show(block=True)

#%%
hist, dw = get_shadow_beam_spatial_distribution(out_beam)

#%%
print(getPeakIntensity(focusing_system))
print(loss_function(focusing_system, 0.))
print(loss_function(focusing_system, -0.005))
print(loss_function(focusing_system, 0.005))
#%%
print(getPeakIntensity(focusing_system))

plot_shadow_beam_spatial_distribution(focusing_system.get_beam())
plt.show(block=True)
