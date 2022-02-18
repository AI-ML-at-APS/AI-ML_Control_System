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

def check_vkb_4(focusing_system):
    # Motor vkb 4
    translations = np.linspace(-0.15, 0.15, 6)
    initial_value = focusing_system.get_vkb_motor_4_translation()

    fig, axes = plt.subplots(2, 3, figsize=[15, 10])

    for ix, trans in enumerate(translations):
        # Moving to a new value
        focusing_system.move_vkb_motor_4_translation(trans, movement=Movement.RELATIVE)
        trans_beam = focusing_system.get_photon_beam(random_seed=DEFAULT_RANDOM_SEED)

        hist, dw = get_shadow_beam_spatial_distribution(trans_beam)
        axes.flat[ix].pcolormesh(hist.hh, hist.vv, hist.data_2D)
        axes.flat[ix].set_title(f'T {trans:3.2f}')

        # Resetting to initial value
        focusing_system.move_vkb_motor_4_translation(initial_value, movement=Movement.ABSOLUTE)
    plt.suptitle('VKB_4')
    plt.tight_layout()
    plt.show(block=True)

def check_hkb_4(focusing_system):
    translations = np.linspace(-0.18, 0.15, 6)
    initial_value = focusing_system.get_hkb_motor_4_translation()

    fig, axes = plt.subplots(2, 3, figsize=[15, 10])

    for ix, trans in enumerate(translations):
        # Moving to a new value
        focusing_system.move_hkb_motor_4_translation(trans, movement=Movement.RELATIVE)
        trans_beam = focusing_system.get_photon_beam(random_seed=DEFAULT_RANDOM_SEED)

        hist, dw = get_shadow_beam_spatial_distribution(trans_beam)
        axes.flat[ix].pcolormesh(hist.hh, hist.vv, hist.data_2D)
        axes.flat[ix].set_title(f'T {trans:3.2f}')

        # Resetting to initial value
        focusing_system.move_hkb_motor_4_translation(initial_value, movement=Movement.ABSOLUTE)
    plt.suptitle('HKB_4')
    plt.tight_layout()
    plt.show(block=True)

def check_vkb_3(focusing_system):
    # Looks like the results for vkb 3 and hkb 3 are suffering from aliasing
    initial_value = focusing_system.get_vkb_motor_3_pitch(units=AngularUnits.DEGREES)

    translations = np.linspace(-0.5, 0.5, 6)

    fig, axes = plt.subplots(2, 3, figsize=[15, 10])

    for ix, trans in enumerate(translations):
        # Moving to a new value
        focusing_system.move_vkb_motor_3_pitch(trans, movement=Movement.RELATIVE, units=AngularUnits.DEGREES)
        trans_beam = focusing_system.get_photon_beam(random_seed=DEFAULT_RANDOM_SEED)

        hist, dw = get_shadow_beam_spatial_distribution(trans_beam)
        axes.flat[ix].pcolormesh(hist.hh, hist.vv, hist.data_2D)
        axes.flat[ix].set_title(f'T {trans:3.2f}')

        # Resetting to initial value
        focusing_system.move_vkb_motor_3_pitch(initial_value, movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES)
    plt.suptitle('VKB_3')
    plt.tight_layout()
    plt.show(block=True)

def check_hkb_3(focusing_system):
    # Looks like the results for vkb 3 and hkb 3 are suffering from aliasing
    initial_value = focusing_system.get_hkb_motor_3_pitch(units=AngularUnits.DEGREES)

    grazing_angles = np.linspace(-0.5, 0.5, 6)

    fig, axes = plt.subplots(2, 3, figsize=[15, 10])

    for ix, angle in enumerate(grazing_angles):
        # Moving to a new value
        focusing_system.move_hkb_motor_3_pitch(angle, movement=Movement.RELATIVE, units=AngularUnits.DEGREES)
        trans_beam = focusing_system.get_photon_beam(random_seed=DEFAULT_RANDOM_SEED)

        hist, dw = get_shadow_beam_spatial_distribution(trans_beam)
        axes.flat[ix].pcolormesh(hist.hh, hist.vv, hist.data_2D)
        axes.flat[ix].set_title(f'T {angle:3.2f}')

        # Resetting to initial value
        focusing_system.move_hkb_motor_3_pitch(initial_value, movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES)
    plt.suptitle('HKB_3')
    plt.tight_layout()
    plt.show(block=True)

def check_vkb_q(focusing_system):
    # Motor vkb q
    # The strange thing about vkb and hkb q distances is that they have minimum values, but not maximum values.
    # For vkb
    # Minimum:
    #     A relative motion of < -200 from the starting position gives an EmptyBeamException.
    # Maximum:
    #     After a certain point (I am not sure exactly when), no matter what "maximum" value I set,
    #     I get the same beam output anyway.

    initial_value = focusing_system.get_vkb_q_distance()

    translations = np.linspace(-200, 300, 6)

    fig, axes = plt.subplots(2, 3, figsize=[15, 10])
    for ix, trans in enumerate(translations):
        # Moving to a new value
        focusing_system.change_vkb_shape(trans, movement=Movement.RELATIVE)
        trans_beam = focusing_system.get_photon_beam(random_seed=DEFAULT_RANDOM_SEED)

        hist, dw = get_shadow_beam_spatial_distribution(trans_beam)
        axes.flat[ix].pcolormesh(hist.hh, hist.vv, hist.data_2D)
        axes.flat[ix].set_title(f'T {trans:3.2f}')

        # Resetting to initial value
        focusing_system.change_vkb_shape(initial_value, movement=Movement.ABSOLUTE)
    plt.suptitle('VKB_Q')
    plt.tight_layout()
    plt.show(block=True)

def check_hkb_q(focusing_system):
    # Motor hkb q
    # The minimum seems to be around -100.
    initial_value = focusing_system.get_hkb_q_distance()

    translations = np.linspace(-100, 300, 6)

    fig, axes = plt.subplots(2, 3, figsize=[15, 10])

    for ix, trans in enumerate(translations):
        # Moving to a new value
        focusing_system.change_hkb_shape(trans, movement=Movement.RELATIVE)
        trans_beam = focusing_system.get_photon_beam(random_seed=DEFAULT_RANDOM_SEED)

        hist, dw = get_shadow_beam_spatial_distribution(trans_beam)
        axes.flat[ix].pcolormesh(hist.hh, hist.vv, hist.data_2D)
        axes.flat[ix].set_title(f'T {trans:3.2f}')

        # Resetting to initial value
        focusing_system.change_hkb_shape(initial_value, movement=Movement.ABSOLUTE)
    plt.suptitle('HKB_Q')
    plt.tight_layout()
    plt.show(block=True)

if __name__=='__main__':
    os.chdir("../work_directory")

    input_beam_path = "primary_optics_system_beam.dat"

    focusing_system = reinitialize(input_beam_path=input_beam_path)
    check_vkb_3(focusing_system)

    focusing_system = reinitialize(input_beam_path)
    check_hkb_3(focusing_system)

    focusing_system = reinitialize(input_beam_path)
    check_vkb_q(focusing_system)

    focusing_system = reinitialize(input_beam_path)
    check_hkb_q(focusing_system)

    clean_up()











