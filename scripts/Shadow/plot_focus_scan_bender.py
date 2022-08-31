import os, numpy

from matplotlib import cm
from matplotlib import pyplot as plt

def plot_3D(xx, yy, zz, label):

    figure = plt.figure(figsize=(10, 7))
    figure.patch.set_facecolor('white')

    axis = figure.add_subplot(111, projection='3d')
    axis.set_zlabel(label + " [mm]")
    axis.set_xlabel("Abs pos up [mm]")
    axis.set_ylabel("Abs pos down [mm]")

    x_to_plot, y_to_plot = numpy.meshgrid(xx, yy)

    axis.plot_surface(x_to_plot, y_to_plot, zz, rstride=1, cstride=1, cmap=cm.autumn, linewidth=0.5, antialiased=True)
    plt.show()


if __name__ == "__main__":
    os.chdir("../../work_directory")

    with open("positions_v.npy", 'rb') as f: positions_v = numpy.load(f)
    with open("positions_h.npy", 'rb') as f: positions_h = numpy.load(f)
    with open("sigma_v.npy", 'rb')     as f: sigma_v     = numpy.load(f)
    with open("fwhm_v.npy", 'rb')      as f: fwhm_v      = numpy.load(f)
    with open("sigma_h.npy", 'rb')     as f: sigma_h     = numpy.load(f)
    with open("fwhm_h.npy", 'rb')      as f: fwhm_h      = numpy.load(f)

    idx = numpy.unravel_index(numpy.argmin(sigma_v[numpy.where(sigma_v > 0.0)]), sigma_v.shape)
    print("V-KB: minimum at: ", positions_v[0, idx[1]], positions_v[1, idx[0]], "Sigma: ", sigma_v[idx]*1e6, "nm")
    idx = numpy.unravel_index(numpy.argmin(sigma_h[numpy.where(sigma_h > 0.0)]), sigma_h.shape)
    print("H-KB: minimum at: ", positions_h[0, idx[1]], positions_h[1, idx[0]], "Sigma: ", sigma_h[idx]*1e6, "nm")

    sigma_v[numpy.where(sigma_v==0.0)] = numpy.nan
    fwhm_v[numpy.where(fwhm_v==0.0)]   = numpy.nan
    sigma_h[numpy.where(sigma_h==0.0)] = numpy.nan
    fwhm_h[numpy.where(fwhm_h==0.0)]   = numpy.nan

    plot_3D(positions_v[0, :], positions_v[1, :], sigma_v.T, "Sigma (V)")
    plot_3D(positions_h[0, :], positions_h[1, :], sigma_h.T, "Sigma (H)")


    #plot_3D(positions_v[0, :], positions_v[1, :], fwhm_v, "FWHM (V)")
    #plot_3D(positions_h[0, :], positions_h[1, :], fwhm_h, "FWHM (H)")

