import os
from pathlib import Path

import numpy as np

import aps
from aps.ai.autoalignment.beamline28IDB.facade.focusing_optics_factory import (
    ExecutionMode, focusing_optics_factory_method)
from aps.ai.autoalignment.beamline28IDB.simulation.facade.focusing_optics_interface import (
    Layout, get_default_input_features)
from aps.ai.autoalignment.common.simulation.facade.parameters import \
    Implementors
from aps.ai.autoalignment.common.util import clean_up
from aps.ai.autoalignment.common.util.common import (AspectRatio, ColorMap,
                                                     PlotMode)
from aps.ai.autoalignment.common.util.shadow.common import (PreProcessorFiles,
                                                            load_shadow_beam)
from aps.ai.autoalignment.common.util.wrappers import \
    get_distribution_info as get_simulated_distribution_info
from aps.ai.autoalignment.common.util.wrappers import plot_distribution

DEFAULT_RANDOM_SEED = np.random.randint(100000)


def setup_work_dir():
    root_dir = Path(aps.__path__[0]).parents[0]
    work_dir = root_dir / "work_directory/28-ID"
    os.chdir(work_dir)


if __name__ == "__main__":

    plot_mode = PlotMode.INTERNAL
    aspect_ratio = AspectRatio.AUTO
    color_map = ColorMap.VIRIDIS

    setup_work_dir()

    input_beam_path = "primary_optics_system_beam.dat"

    clean_up()

    # Initializing the focused beam from simulation
    input_features = get_default_input_features(layout=Layout.AUTO_ALIGNMENT)
    input_beam = load_shadow_beam(input_beam_path)
    focusing_system = focusing_optics_factory_method(
        execution_mode=ExecutionMode.SIMULATION,
        implementor=Implementors.SHADOW,
        bender=True,
    )

    focusing_system.initialize(
        input_photon_beam=input_beam,
        rewrite_preprocessor_files=PreProcessorFiles.NO,
        layout=Layout.AUTO_ALIGNMENT,
        input_features=input_features,
    )

    beam = focusing_system.get_photon_beam(random_seed=DEFAULT_RANDOM_SEED)
    hist, dw = get_simulated_distribution_info(implementor=Implementors.SHADOW, beam=beam)
    print(dw)

    plot_distribution(
        implementor=Implementors.SHADOW,
        beam=beam,
        title="Initial Beam",
        plot_mode=plot_mode,
        aspect_ratio=aspect_ratio,
        color_map=color_map,
    )
    clean_up()
