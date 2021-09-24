#
# Python script to run shadow3. Created automatically with ShadowTools.make_python_script_from_list().
#
import Shadow
import numpy
import os, glob, sys

from oasys.util.oasys_util import get_sigma, get_fwhm, get_average

from orangecontrib.ml.util.mocks import MockWidget
from orangecontrib.ml.util.data_structures import ListOfParameters, DictionaryWrapper

# OASYS + HYBRID library, to add correction for diffraction and error profiles interference effects.
from orangecontrib.shadow.util.shadow_objects import ShadowBeam, ShadowSource, ShadowOpticalElement
from orangecontrib.shadow.util.hybrid         import hybrid_control

from orangecontrib.shadow_advanced_tools.widgets.sources.ow_hybrid_undulator import HybridUndulatorAttributes
import orangecontrib.shadow_advanced_tools.widgets.sources.bl.hybrid_undulator_bl as hybrid_undulator_bl

def get_hybrid_input_parameters(shadow_beam, diffraction_plane=1, calcType=1, nf=0, verbose=False):
    input_parameters = hybrid_control.HybridInputParameters()
    input_parameters.ghy_lengthunit = 2
    input_parameters.widget = MockWidget(verbose=verbose)
    input_parameters.shadow_beam = shadow_beam
    input_parameters.ghy_diff_plane = diffraction_plane
    input_parameters.ghy_calcType = calcType
    input_parameters.ghy_distance = -1
    input_parameters.ghy_focallength = -1
    input_parameters.ghy_nf = nf
    input_parameters.ghy_nbins_x = 100
    input_parameters.ghy_nbins_z = 100
    input_parameters.ghy_npeak = 10
    input_parameters.ghy_fftnpts = 10000
    input_parameters.file_to_write_out = 0
    input_parameters.ghy_automatic = 0

    return input_parameters

def clean_up():
    for pattern in ["angle.*", "effic.*", "mirr.*", "optax.*", "rmir.*", "screen.*", "star.*"]:
        files = glob.glob(os.path.join(os.curdir, pattern), recursive=True)
        for file in files: os.remove(file)

class MockUndulatorHybrid(MockWidget, HybridUndulatorAttributes):
    def __init__(self, verbose=False,  workspace_units=2):
        MockWidget.__init__(self, verbose, workspace_units)

        self.distribution_source = 0  # SRW
        self.optimize_source = 0
        self.polarization = 1
        self.coherent_beam = 0
        self.phase_diff = 0.0
        self.polarization_degree = 1.0
        self.max_number_of_rejected_rays = 0

        self.use_harmonic = 2
        self.energy = 4999
        self.energy_to = 5001
        self.energy_points = 3#21

        self.number_of_periods = 72  # Number of ID Periods (without counting for terminations
        self.undulator_period = 0.033  # Period Length [m]
        self.horizontal_central_position = 0.0
        self.vertical_central_position = 0.0
        self.longitudinal_central_position = 0.0

        self.Kv = 1.907944
        self.Kh = 0.0
        self.magnetic_field_from = 0
        self.initial_phase_vertical = 0.0
        self.initial_phase_horizontal = 0.0
        self.symmetry_vs_longitudinal_position_vertical = 1
        self.symmetry_vs_longitudinal_position_horizontal = 0

        self.electron_energy_in_GeV = 7.0
        self.electron_energy_spread = 0.00098
        self.ring_current = 0.1
        self.electron_beam_size_h = 0.0002805
        self.electron_beam_size_v = 1.02e-05
        self.electron_beam_divergence_h = 1.18e-05
        self.electron_beam_divergence_v = 3.4e-06

        self.type_of_initialization = 0

        self.source_dimension_wf_h_slit_gap = 0.005
        self.source_dimension_wf_v_slit_gap = 0.001
        self.source_dimension_wf_h_slit_points = 500
        self.source_dimension_wf_v_slit_points = 100
        self.source_dimension_wf_distance = 10.0

        self.horizontal_range_modification_factor_at_resizing = 0.5
        self.horizontal_resolution_modification_factor_at_resizing = 5.0
        self.vertical_range_modification_factor_at_resizing = 0.5
        self.vertical_resolution_modification_factor_at_resizing = 5.0

        self.auto_expand = 0
        self.auto_expand_rays = 0

        self.kind_of_sampler = 1
        self.save_srw_result = 0

def run_hybrid_undulator_source(n_rays=500000, random_seed=5676561):
    widget = MockUndulatorHybrid(verbose=True)

    widget.number_of_rays = n_rays
    widget.seed = random_seed

    source_beam, _ = hybrid_undulator_bl.run_hybrid_undulator_simulation(widget)

    return source_beam

def run_geometrical_source(n_rays=500000, random_seed=5676561):
    #####################################################
    # SHADOW 3 INITIALIZATION

    #
    # initialize shadow3 source (oe0) and beam
    #
    oe0 = Shadow.Source()

    oe0.FDISTR = 3
    oe0.F_COLOR = 3
    oe0.F_PHOT = 0
    oe0.HDIV1 = 4e-07
    oe0.HDIV2 = 4e-07
    oe0.IDO_VX = 0
    oe0.IDO_VZ = 0
    oe0.IDO_X_S = 0
    oe0.IDO_Y_S = 0
    oe0.IDO_Z_S = 0
    oe0.ISTAR1 = random_seed
    oe0.NPOINT = n_rays
    oe0.PH1 = 4999.0
    oe0.PH2 = 5001.0
    oe0.SIGDIX = 1.6e-05
    oe0.SIGDIZ = 7e-06
    oe0.SIGMAX = 0.281
    oe0.SIGMAZ = 0.014
    oe0.VDIV1 = 8e-07
    oe0.VDIV2 = 8e-07

    # WEIRD MEMORY INITIALIZATION BY FORTRAN. JUST A FIX.
    def fix_Intensity(beam_out, polarization=0):
        if polarization == 0:
            beam_out._beam.rays[:, 15] = 0
            beam_out._beam.rays[:, 16] = 0
            beam_out._beam.rays[:, 17] = 0
        return beam_out

    shadow_source = ShadowSource.create_src()
    shadow_source.set_src(src=oe0)

    # Run SHADOW to create the source + BUT WE USE OASYS LIBRARY TO BE ABLE TO RUN HYBRYD
    source_beam = fix_Intensity(ShadowBeam.traceFromSource(shadow_source))

    return source_beam

def run_invariant_shadow_simulation(source_beam):
    #####################################################
    # SHADOW 3 INITIALIZATION

    #
    # initialize shadow3 source (oe0) and beam
    #
    oe1 = Shadow.OE()
    oe2 = Shadow.OE()
    oe3 = Shadow.OE()
    oe4 = Shadow.OE()

    #
    # Define variables. See meaning of variables in:
    #  https://raw.githubusercontent.com/srio/shadow3/master/docs/source.nml
    #  https://raw.githubusercontent.com/srio/shadow3/master/docs/oe.nml
    #


    # WB SLITS
    oe1.DUMMY = 0.1
    oe1.FWRITE = 3
    oe1.F_REFRAC = 2
    oe1.F_SCREEN = 1
    oe1.I_SLIT = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    oe1.N_SCREEN = 1
    oe1.RX_SLIT = numpy.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    oe1.RZ_SLIT = numpy.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    oe1.T_IMAGE = 0.0
    oe1.T_INCIDENCE = 0.0
    oe1.T_REFLECTION = 180.0
    oe1.T_SOURCE = 26800.0

    # PLANE MIRROR
    oe2.DUMMY = 0.1
    oe2.FHIT_C = 1
    oe2.FILE_REFL = b'Pt.dat'
    oe2.FWRITE = 1
    oe2.F_REFLEC = 1
    oe2.RLEN1 = 250.0
    oe2.RLEN2 = 250.0
    oe2.RWIDX1 = 10.0
    oe2.RWIDX2 = 10.0
    oe2.T_IMAGE = 0.0
    oe2.T_INCIDENCE = 89.7135211024
    oe2.T_REFLECTION = 89.7135211024
    oe2.T_SOURCE = 2800.0

    # DCM-1
    oe3.DUMMY = 0.1
    oe3.FHIT_C = 1
    oe3.FILE_REFL = b'Si111.dat'
    oe3.FWRITE = 1
    oe3.F_CENTRAL = 1
    oe3.F_CRYSTAL = 1
    oe3.PHOT_CENT = 5000.0
    oe3.RLEN1 = 50.0
    oe3.RLEN2 = 50.0
    oe3.RWIDX1 = 50.0
    oe3.RWIDX2 = 50.0
    oe3.R_LAMBDA = 5000.0
    oe3.T_IMAGE = 0.0
    oe3.T_INCIDENCE = 66.7041090078
    oe3.T_REFLECTION = 66.7041090078
    oe3.T_SOURCE = 15400.0

    # DCM-2
    oe4.ALPHA = 180.0
    oe4.DUMMY = 0.1
    oe4.FHIT_C = 1
    oe4.FILE_REFL = b'Si111.dat'
    oe4.FWRITE = 1
    oe4.F_CENTRAL = 1
    oe4.F_CRYSTAL = 1
    oe4.PHOT_CENT = 5000.0
    oe4.RLEN1 = 50.0
    oe4.RLEN2 = 50.0
    oe4.RWIDX1 = 50.0
    oe4.RWIDX2 = 50.0
    oe4.R_LAMBDA = 5000.0
    oe4.T_IMAGE = 5494.324
    oe4.T_INCIDENCE = 66.7041090078
    oe4.T_REFLECTION = 66.7041090078
    oe4.T_SOURCE = 8.259

    output_beam = ShadowBeam.traceFromOE(source_beam.duplicate(), ShadowOpticalElement(oe1), widget_class_name="ScreenSlits")
    output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), ShadowOpticalElement(oe2), widget_class_name="PlaneMirror")
    output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), ShadowOpticalElement(oe3), widget_class_name="PlaneCrystal")
    output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), ShadowOpticalElement(oe4), widget_class_name="PlaneCrystal")

    return output_beam

def run_ML_shadow_simulation(input_beam, input_features=DictionaryWrapper(), plot=False, verbose=False):
    oe5 = Shadow.OE()
    oe6 = Shadow.OE()
    oe7 = Shadow.OE()
    oe8 = Shadow.OE()

    # COHERENCE SLITS
    oe5.DUMMY = 0.1
    oe5.FWRITE = 3
    oe5.F_REFRAC = 2
    oe5.F_SCREEN = 1
    oe5.I_SLIT = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    oe5.N_SCREEN = 1
    oe5.CX_SLIT = numpy.array([input_features.get_parameter("coh_slits_h_center"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    oe5.CZ_SLIT = numpy.array([input_features.get_parameter("coh_slits_v_center"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    oe5.RX_SLIT = numpy.array([input_features.get_parameter("coh_slits_h_aperture"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    oe5.RZ_SLIT = numpy.array([input_features.get_parameter("coh_slits_v_aperture"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    oe5.T_IMAGE = 0.0
    oe5.T_INCIDENCE = 0.0
    oe5.T_REFLECTION = 180.0
    oe5.T_SOURCE = 0.0

    # V-KB
    oe6.ALPHA = 180.0
    oe6.DUMMY = 0.1
    oe6.FCYL = 1
    oe6.FHIT_C = 1
    oe6.FILE_REFL = b'Pt.dat'
    oe6.FILE_RIP = b'VKB.dat'
    oe6.FMIRR = 2
    oe6.FWRITE = 1
    oe6.F_DEFAULT = 0
    oe6.F_G_S = 2
    oe6.F_REFLEC = 1
    oe6.F_RIPPLE = 1

    oe6.RLEN1 = 50.0
    oe6.RLEN2 = 50.0
    oe6.RWIDX1 = 10.0
    oe6.RWIDX2 = 10.0
    oe6.SIMAG = input_features.get_parameter("vkb_p_distance")
    oe6.SSOUR = 50667.983
    oe6.THETA = 89.8281126615
    oe6.T_IMAGE = 101.0
    oe6.T_INCIDENCE = 89.8281126615
    oe6.T_REFLECTION = 89.8281126615
    oe6.T_SOURCE = 150.0

    # DISPLACEMENT
    oe6.F_MOVE = 1
    oe6.OFFX =  input_features.get_parameter("vkb_offset_x")
    oe6.OFFY =  input_features.get_parameter("vkb_offset_y")
    oe6.OFFZ =  input_features.get_parameter("vkb_offset_z")
    oe6.X_ROT = input_features.get_parameter("vkb_rotation_x")
    oe6.Y_ROT = input_features.get_parameter("vkb_rotation_y")
    oe6.Z_ROT = input_features.get_parameter("vkb_rotation_z")

    # H-KB
    oe7.ALPHA = 90.0
    oe7.DUMMY = 0.1
    oe7.FCYL = 1
    oe7.FHIT_C = 1
    oe7.FILE_REFL = b'Pt.dat'
    oe7.FILE_RIP = b'HKB.dat'
    oe7.FMIRR = 2
    oe7.FWRITE = 1
    oe7.F_DEFAULT = 0
    oe7.F_G_S = 2
    oe7.F_MOVE = 1
    oe7.F_REFLEC = 1
    oe7.F_RIPPLE = 1
    oe7.RLEN1 = 50.0
    oe7.RLEN2 = 50.0
    oe7.RWIDX1 = 10.0
    oe7.RWIDX2 = 10.0
    oe7.SIMAG = input_features.get_parameter("hkb_p_distance")
    oe7.SSOUR = 50768.983
    oe7.THETA = 89.8281126615
    oe7.T_IMAGE = 120.0
    oe7.T_INCIDENCE = 89.8281126615
    oe7.T_REFLECTION = 89.8281126615
    oe7.T_SOURCE = 0.0

    # DISPLACEMENT
    oe7.F_MOVE = 1
    oe7.OFFX =  input_features.get_parameter("hkb_offset_x")
    oe7.OFFY =  input_features.get_parameter("hkb_offset_y")
    oe7.OFFZ =  input_features.get_parameter("hkb_offset_z")
    oe7.X_ROT = input_features.get_parameter("hkb_rotation_x")
    oe7.Y_ROT = input_features.get_parameter("hkb_rotation_y")
    oe7.Z_ROT = input_features.get_parameter("hkb_rotation_z")

    oe8.ALPHA = 270.0
    oe8.DUMMY = 0.1
    oe8.FWRITE = 3
    oe8.F_REFRAC = 2
    oe8.T_IMAGE = 0.0
    oe8.T_INCIDENCE = 0.0
    oe8.T_REFLECTION = 180.0
    oe8.T_SOURCE = 0.0

    # HYBRID CORRECTION TO CONSIDER DIFFRACTION FROM SLITS
    output_beam = ShadowBeam.traceFromOE(input_beam.duplicate(), ShadowOpticalElement(oe5), widget_class_name="ScreenSlits")
    output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                    diffraction_plane=4,  # BOTH 1D+1D (3 is 2D)
                                                                    calcType=1,  # Diffraction by Simple Aperture
                                                                    verbose=verbose)).ff_beam
    # HYBRID CORRECTION TO CONSIDER MIRROR SIZE AND ERROR PROFILE
    output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), ShadowOpticalElement(oe6), widget_class_name="EllypticalMirror")
    output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                    diffraction_plane=1, # Tangential
                                                                    calcType=3, # Diffraction by Mirror Size + Errors
                                                                    nf=1,
                                                                    verbose=verbose)).nf_beam
    
    # HYBRID CORRECTION TO CONSIDER MIRROR SIZE AND ERROR PROFILE
    output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), ShadowOpticalElement(oe7), widget_class_name="EllypticalMirror")
    output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                    diffraction_plane=1, # Tangential
                                                                    calcType=3, # Diffraction by Mirror Size + Errors
                                                                    nf=1,
                                                                    verbose=verbose)).nf_beam
    
    output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), ShadowOpticalElement(oe8), widget_class_name="EmptyElement")
    
    to_micron = 1e3
    to_urad = 1e6
    
    output_beam._beam.rays[:, 0] *= to_micron
    output_beam._beam.rays[:, 1] *= to_micron
    output_beam._beam.rays[:, 2] *= to_micron
    
    output_beam._beam.rays[:, 3] *= to_urad
    output_beam._beam.rays[:, 4] *= to_urad
    output_beam._beam.rays[:, 5] *= to_urad

    if plot:
        ticket = Shadow.ShadowTools.plotxy(output_beam._beam, 1, 3, nbins=201, nolost=1, title="Focal Spot Size", xrange=[-2.0, +2.0], yrange=[-2.0, +2.0])
    else:
        ticket = output_beam._beam.histo2(1, 3, nbins=201, nolost=1, xrange=[-2.0, +2.0], yrange=[-2.0, +2.0])

    ticket['fwhm_h'], ticket['fwhm_quote_h'], ticket['fwhm_coordinates_h'] = get_fwhm(ticket['histogram_h'], ticket['bin_h_center'])
    ticket['fwhm_v'], ticket['fwhm_quote_v'], ticket['fwhm_coordinates_v'] = get_fwhm(ticket['histogram_v'], ticket['bin_v_center'])
    ticket['sigma_h']    = get_sigma(ticket['histogram_h'], ticket['bin_h_center'])
    ticket['sigma_v']    = get_sigma(ticket['histogram_v'], ticket['bin_v_center'])
    ticket['centroid_h'] = get_average(ticket['histogram_h'], ticket['bin_h_center'])
    ticket['centroid_v'] = get_average(ticket['histogram_v'], ticket['bin_v_center'])
    
    histogram = ticket["histogram"]
    
    peak_intensity     = numpy.average(histogram[numpy.where(histogram >= numpy.max(histogram) * 0.90)])
    integral_intensity = numpy.sum(histogram)

    output_parameters = DictionaryWrapper(
        h_sigma=ticket['sigma_h'],
        h_fwhm=ticket['fwhm_h'],
        h_centroid=ticket['centroid_h'],
        v_sigma=ticket['sigma_v'],
        v_fwhm=ticket['fwhm_v'],
        v_centroid=ticket['centroid_v'],
        integral_intensity=integral_intensity,
        peak_intensity=peak_intensity
    )
    
    if plot:
        ticket = Shadow.ShadowTools.plotxy(output_beam._beam, 4, 6, nbins=201, nolost=1, title="Focal Spot Divergence", xrange=[-150.0, +150.0], yrange=[-200.0, +200.0])
    else:
        ticket = output_beam._beam.histo2(4, 6, nbins=201, nolost=1, xrange=[-150.0, +150.0], yrange=[-200.0, +200.0])
    
    output_parameters.set_parameter("h_divergence", get_average(ticket['histogram_h'], ticket['bin_h_center']))
    output_parameters.set_parameter("v_divergence", get_average(ticket['histogram_v'], ticket['bin_v_center']))

    return output_parameters

from oasys.util.oasys_util import TTYGrabber

if __name__=="__main__":
    clean_up()

    #source_beam = run_geometrical_source(n_rays=50000)
    source_beam = run_hybrid_undulator_source(n_rays=5000000)

    input_beam = run_invariant_shadow_simulation(source_beam)

    output_parameters = run_ML_shadow_simulation(input_beam,
                                                 input_features=DictionaryWrapper(coh_slits_h_aperture = 0.03,
                                                                                            coh_slits_h_center = 0.0,
                                                                                            coh_slits_v_aperture = 0.07,
                                                                                            coh_slits_v_center = 0.0,
                                                                                            vkb_p_distance=221,
                                                                                            vkb_offset_x  = 0.0,
                                                                                            vkb_offset_y  =0.0,
                                                                                            vkb_offset_z  =0.0,
                                                                                            vkb_rotation_x=0.0,
                                                                                            vkb_rotation_y=0.0,
                                                                                            vkb_rotation_z=0.0,
                                                                                            hkb_p_distance=120,
                                                                                            hkb_offset_x = 0.0,
                                                                                            hkb_offset_y=0.0,
                                                                                            hkb_offset_z=0.0,
                                                                                            hkb_rotation_x=0.0,
                                                                                            hkb_rotation_y=0.0,
                                                                                            hkb_rotation_z=0.0),
                                                 verbose=True, plot=True)

    clean_up()

    '''
    show_shadow = True

    class DontPrint(object):
        def write(*args): pass

    ttx = TTYGrabber()

    # create a list of possible values to map the change in curvature of both the kb mirrors
    # FROM OASYS p value change in range +-20 mm with step 1 (visual scanning loop)

    #vkb_p_distances = numpy.arange(start=201.0, stop=242.0, step=1.0)
    #hkb_p_distances = numpy.arange(start=100.0, stop=141.0, step=1.0)
    vkb_p_distances = numpy.arange(start=219.0, stop=224.0, step=1.0)
    hkb_p_distances = numpy.arange(start=118.0, stop=123.0, step=1.0)

    input_features_list = ListOfParameters()

    for vkb_p_distance in vkb_p_distances:
        for hkb_p_distance  in hkb_p_distances:
            input_features_list.add_parameters(DictionaryWrapper(coh_slits_h_aperture = 0.03,
                                                                 coh_slits_h_center = 0.0,
                                                                 coh_slits_v_aperture = 0.07,
                                                                 coh_slits_v_center = 0.0,
                                                                 vkb_p_distance=vkb_p_distance,
                                                                 vkb_offset_x  = 0.0,
                                                                 vkb_offset_y  =0.0,
                                                                 vkb_offset_z  =0.0,
                                                                 vkb_rotation_x=0.0,
                                                                 vkb_rotation_y=0.0,
                                                                 vkb_rotation_z=0.0,
                                                                 hkb_p_distance=hkb_p_distance,
                                                                 hkb_offset_x = 0.0,
                                                                 hkb_offset_y=0.0,
                                                                 hkb_offset_z=0.0,
                                                                 hkb_rotation_x=0.0,
                                                                 hkb_rotation_y=0.0,
                                                                 hkb_rotation_z=0.0))

    # store the input file
    input_features_list.to_npy_file("input")

    # load the input file
    input_features_list = ListOfParameters()
    input_features_list.from_npy_file("input.npy")

    output_parameters_list = ListOfParameters()

    if not show_shadow: ttx.start()

    input_beam = run_invariant_shadow_simulation(n_rays=5000)

    try:
        for index in range(input_features_list.get_number_of_parameters()):
            output_parameters = run_ML_shadow_simulation(input_beam, input_features=input_features_list.get_parameters(index))
            output_parameters_list.add_parameters(output_parameters)
            print("Run simulation # ", index+1)
    except Exception as e:
        print(e)

    if not show_shadow: ttx.stop()

    clean_up()

    output_parameters_list.to_npy_file("output.npy")

    print("*************************************************")
    print("*************************************************")
    print("              INPUT/OUTPUT")
    print("*************************************************")
    print("*************************************************")

    print(input_features_list.get_number_of_parameters())
    print(output_parameters_list.get_number_of_parameters())

    for index in range(input_features_list.get_number_of_parameters()):
        print("------- Input:")
        print(input_features_list.get_parameters(index))
        print("        Output:")
        print(output_parameters_list.get_parameters(index))

    #####################
    # SAVE/LOAD AS TRAINING DATA

    result = numpy.full((1, 2), None)
    result[0, 0] = input_features_list.to_numpy_matrix()
    result[0, 1] = output_parameters_list.to_numpy_matrix()

    numpy.save("database.npy", result, allow_pickle=True)

    test = numpy.load("database.npy", allow_pickle=True)

    input_features_list = ListOfParameters().from_numpy_matrix(test[0, 0])
    output_parameters_list = ListOfParameters().from_numpy_matrix(test[0, 1])

    print(test)
    '''
