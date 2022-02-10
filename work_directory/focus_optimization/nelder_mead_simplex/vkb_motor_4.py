#%%
import os
from beamline34IDC.simulation.facade import Implementors
from beamline34IDC.simulation.facade.focusing_optics_factory import focusing_optics_factory_method
from beamline34IDC.simulation.facade.focusing_optics_interface import Movement

from beamline34IDC.util.shadow.common import \
    plot_shadow_beam_spatial_distribution, get_shadow_beam_spatial_distribution,\
    load_shadow_beam, PreProcessorFiles
from beamline34IDC.util import clean_up
import matplotlib.pyplot as plt

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

def reinitialize():
    focusing_system = focusing_optics_factory_method(implementor=Implementors.SHADOW)

    focusing_system.initialize(input_photon_beam=input_beam,
                               rewrite_preprocessor_files=PreProcessorFiles.NO,
                               rewrite_height_error_profile_files=False)
    output_beam = focusing_system.get_photon_beam(random_seed=101, remove_lost_rays=True)
    return focusing_system, output_beam

def runs(focusing_system, n_runs=5, translation=None,
         translation_type=Movement.RELATIVE,
         fig_save_fname=None):
    hists_all = []
    dws_all = []

    n_rows = n_runs // 5 + 1
    n_cols = 5

    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=[10, 2 * n_rows])
    #fig2, axes2 = plt.subplot()

    for i in range(n_runs):
        if translation is not None:
            focusing_system.move_vkb_motor_4_translation(translation, translation_type)
        out_beam = focusing_system.get_photon_beam(remove_lost_rays=True)
        hist, dw = get_shadow_beam_spatial_distribution(out_beam)
        hists_all.append(hist)
        dws_all.append(dw)

        ax = axes1[i // 5, i % 5]
        ax.pcolormesh(hist. hh, hist.vv, hist.data_2D)
        ax.set_aspect('equal')

        #h_sigma = ticket['sigma_h'],
        #h_fwhm = ticket['fwhm_h'],
        #h_centroid = ticket['centroid_h'],
        #v_sigma = ticket['sigma_v'],
        #v_fwhm = ticket['fwhm_v'],
        #v_centroid = ticket['centroid_v'],
        #integral_intensity = integral_intensity,
        #peak_intensity = peak_intensity
    fig1.tight_layout()
    if fig_save_fname is not None:
        fig1.savefig(fig_save_fname, bbox_inches='tight')
    plt.show(block=True)
    return hists_all, dws_all

#%%
work_dir = '/Users/saugat/code/oasys/ML_Control_System/work_directory'
os.chdir(work_dir)
input_beam = load_shadow_beam("primary_optics_system_beam.dat")

#%%
# Focusing Optics System -------------------------

focusing_system, out_beam = reinitialize()

#%%
plot_shadow_beam_spatial_distribution(out_beam)
plt.show(block=True)

#%%
hist, dw = get_shadow_beam_spatial_distribution(out_beam, do_gaussian_fit=True)

#%%
print(getPeakIntensity(focusing_system))
print(loss_function(focusing_system, 0.))
print(loss_function(focusing_system, -0.005))
print(loss_function(focusing_system, 0.005))
#%%
print(getPeakIntensity(focusing_system))

plot_shadow_beam_spatial_distribution(focusing_system.get_beam())
plt.show(block=True)
