import os

from aps.ai.autoalignment.common.simulation.facade.parameters import Implementors
from aps.ai.autoalignment.beamline34IDC.facade.focusing_optics_factory import focusing_optics_factory_method, ExecutionMode
from aps.ai.autoalignment.beamline34IDC.facade.focusing_optics_interface import Movement, AngularUnits, DistanceUnits
from aps.ai.autoalignment.common.util.shadow.common import plot_shadow_beam_spatial_distribution, load_shadow_beam, PreProcessorFiles
from aps.ai.autoalignment.common.util import clean_up

if __name__ == "__main__":
    verbose = False

    os.chdir("../work_directory")

    clean_up()

    input_beam = load_shadow_beam("primary_optics_system_beam.dat")

    # Focusing Optics System -------------------------

    focusing_system = focusing_optics_factory_method(execution_mode=ExecutionMode.SIMULATION,
                                                     implementor=Implementors.SHADOW, bender=True)

    focusing_system.initialize(input_photon_beam=input_beam,
                               rewrite_preprocessor_files=PreProcessorFiles.NO,
                               rewrite_height_error_profile_files=False)

    #print("Initial V-KB bender positions and q (up, down) ",
    #      focusing_system.get_vkb_motor_1_2_bender(units=DistanceUnits.MICRON), focusing_system.get_vkb_q_distance())
    #print("Initial H-KB bender positions and q (up, down)",
    #      focusing_system.get_hkb_motor_1_2_bender(units=DistanceUnits.MICRON), focusing_system.get_hkb_q_distance())

    for i in range(50):
        focusing_system.move_vkb_motor_3_pitch(0.00001, movement=Movement.RELATIVE, units=AngularUnits.MILLIRADIANS)
        output_beam = focusing_system.get_photon_beam(verbose=verbose, near_field_calculation=False, debug_mode=False,
                                                      random_seed=2120)
        print('iteration', i)

    clean_up()
