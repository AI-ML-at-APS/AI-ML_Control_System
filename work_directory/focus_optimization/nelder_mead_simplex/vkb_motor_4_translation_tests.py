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
import numpy as np
from beamline34IDC.util.redirect_shadow_output import redirected_output

#%%

def reinitialize():
    clean_up()
    focusing_system = focusing_optics_factory_method(implementor=Implementors.SHADOW)

    focusing_system.initialize(input_photon_beam=input_beam,
                               rewrite_preprocessor_files=PreProcessorFiles.NO,
                               rewrite_height_error_profile_files=False)
    with redirected_output():
        output_beam = focusing_system.get_photon_beam(random_seed=101, remove_lost_rays=True)
    return focusing_system, output_beam

def runs(n_runs=3, translation=None,
         translation_type=Movement.RELATIVE,
         fig_save_prefix=None,
         random_seed=None):
    global focusing_system
    hists_all = []
    dws_all = []

    n_rows = (n_runs + 1) // 3
    n_cols = 3

    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=[10, 4 * n_rows], constrained_layout=True)
    fig2, axes2 = plt.subplots(1, 3, figsize=[9,3], constrained_layout=True)

    axes1 = axes1.flatten()
    centroids_h = []
    centroids_v = []
    fwhms_h = []
    fwhms_v = []
    peak_intensities = []


    for i in range(n_runs):
        if translation is not None:
            if np.ndim(translation) == 0:
                translation_this = translation
            elif np.ndim(translation) == 1 and np.size(translation) == n_runs:
                translation_this = translation[i]
            else:
                raise ValueError
            focusing_system.move_vkb_motor_4_translation(translation_this, translation_type)
        with redirected_output():
            out_beam = focusing_system.get_photon_beam(remove_lost_rays=True, random_seed=random_seed)
        hist, dw = get_shadow_beam_spatial_distribution(out_beam)
        hists_all.append(hist)
        dws_all.append(dw)

        ax = axes1[i]
        ax.pcolormesh(hist. hh, hist.vv, hist.data_2D)
        ax.set_aspect('equal')

        centroids_v.append(dw.get_parameter('v_centroid'))
        centroids_h.append(dw.get_parameter('h_centroid'))
        fwhms_h.append(dw.get_parameter('h_fwhm'))
        fwhms_v.append(dw.get_parameter('v_fwhm'))
        peak_intensities.append(dw.get_parameter('peak_intensity'))

    axes2[0].plot(centroids_h, centroids_v, marker='o')
    axes2[0].set_xlabel('Centroid_h')
    axes2[0].set_ylabel('Centriod_v')
    axes2[1].plot(fwhms_h, fwhms_v, marker='o')
    axes2[1].set_xlabel('fwhm_h')
    axes2[1].set_ylabel('fwhm_v')
    axes2[2].plot(peak_intensities, marker='o')
    axes2[2].set_ylabel("peak intensity")

    trans_string = str(translation)
    fig1.suptitle(f'trans {trans_string}')
    fig2.suptitle(f'trans {trans_string}')
    if fig_save_prefix is not None:
        name1 = f'{fig_save_prefix}_distribs.png'
        name2 = f'{fig_save_prefix}_metrics.png'
        fig1.savefig(name1, bbox_inches='tight')
        fig2.savefig(name2, bbox_inches='tight')
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
hist1, dws1 = runs(5, fig_save_prefix='figures/no_trans_no_seed')

#%%

hists2, dws2 = runs( 3, random_seed=101, fig_save_prefix='figures/no_trans_seed_101')
hists3, dws3 = runs( 3, random_seed=1, fig_save_prefix='figures/no_trans_seed_1')
hists4, dws4 = runs( 3, random_seed=1, translation=0, fig_save_prefix='figures/trans_0_seed_1')

#%%
# Since I am using the same random seed (1) in the last two runs without translation and this run with some
# translation, I get identical peak intensities for dw3, dw4, and dw5.
hists5, dws5 = runs( 5, translation=[0, -0.05, 0.025, 0.025, 0], random_seed=1,
                    fig_save_prefix='figures/trans_seed_1')

#%%
print(dws4[2].get_parameter('peak_intensity'),
      dws5[0].get_parameter('peak_intensity'),
      dws5[4].get_parameter('peak_intensity'))

#%%
# Now I am going run an experiment that changes the random seed, then go back to seed 1.
hists6, dws6 = runs( 3, random_seed=101, fig_save_prefix='figures/no_trans_seed_101_2')
hists7, dws7 = runs(3, random_seed=1, fig_save_prefix='figures/no_trans_seed_1_2')

#%%
# Now I am going run an experiment that changes the random seed, then go back to seed 1 WITH translations.
hists8, dws8 = runs(3, random_seed=101, fig_save_prefix='figures/no_trans_seed_101_3')
hists9, dws9 = runs(5, random_seed=1, translation=[0, -0.05, 0.025, 0.025, 0],
                    fig_save_prefix='figures/trans_seed_1_3')
#%%
# These should be the same, but they produce different values. There's some oddity (or memory leak?) going on somewhere
print(dws7[0].get_parameter('peak_intensity'), dws9[0].get_parameter('peak_intensity'))

#%%
print(getPeakIntensity(focusing_system))
print(loss_function(focusing_system, 0.))
print(loss_function(focusing_system, -0.005))
print(loss_function(focusing_system, 0.005))
#%%
print(getPeakIntensity(focusing_system))

plot_shadow_beam_spatial_distribution(focusing_system.get_beam())
plt.show(block=True)
