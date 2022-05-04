import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution

def generalized_1D_gaussian(xdata_tuple: np.ndarray,  # array consisting of 21d points.
                            amplitude: float,
                            center_x: float,
                            sigma_x: float,
                            offset: float) -> np.ndarray:

    return offset + amplitude * np.exp(-(((xdata_tuple - center_x) ** 2)/ (2 * sigma_x ** 2) ))


def generalized_2D_gaussian(xdata_tuple: np.ndarray,  # array consisting of 2d points.
                            amplitude: float,
                            center_x: float,
                            center_y: float,
                            sigma_x: float,
                            sigma_y: float,
                            theta: float,
                            offset: float) -> np.ndarray:

    XX = xdata_tuple[:,0]
    YY = xdata_tuple[:,1]

    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(-(a * ((XX - center_x) ** 2)
                                      + 2 * b * (XX - center_x) * (YY -center_y)
                                      + c * ((YY - center_y) ** 2)))
    return g

def calculate_1D_gaussian_fit(data_1D: np.ndarray, x: np.ndarray = None) -> dict:
    if x is None :
        nx = len(data_1D)
        x = np.arange(-nx // 2, nx // 2)

    bounds_min = [0.0,           x[0],  0.0,       0.0]
    bounds_max = [data_1D.sum(), x[-1], x[-1] * 2, data_1D.max()]
    popt, _ = curve_fit(generalized_1D_gaussian,
                        x,
                        data_1D,
                        bounds=[bounds_min, bounds_max])
    amplitude, center_x, sigma_x, offset = popt

    gaussian_fit = {"amplitude": amplitude,
                    "center_x": center_x,
                    "sigma_x": sigma_x,
                    "fwhm_x": 2.355 * sigma_x,
                    "offset": offset}

    return gaussian_fit



def calculate_2D_gaussian_fit(data_2D: np.ndarray, x: np.ndarray = None, y: np.ndarray = None) -> dict:
    r"""Fit a 2-d gaussian to the probe intensities (not amplitudes) and return the fit parameters.

    The returned dictionary contains the following fit parameters (as described in [1]_):
        * ``amplitude`` : Amplitude of the fitted gaussian.
        * ``center_x`` : X-offset of the center of the fitted gaussian.
        * ``center_y`` : Y-offset of the center of the fitted gaussian.
        * ``theta`` : Clockwise rotation angle for the gaussian fit. Prior to the rotation, primary axes of the \
          | gaussian  are aligned to the X and Y axes.
        * ``sigma_x`` : Spread (standard deviation) of the gaussian fit along the x-axis (prior to rotation).
        * ``fwhm_x`` : FWHM along x-axis (prior to rotation).
        * ``sigma_y`` : Spread of the gaussian fit along the y-axis (prior to rotation).
        * ``fwhm_y`` : FWHM along y-axis (before rotation).
        * ``offset`` : Constant level of offset applied to the fit.

    Returns
    -------
    out : dict
        Dictionary containing the fit parameters.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    """

    if x is None and y is None:
        nx = data_2D.shape[-1]
        ny = data_2D.shape[-2]
        y = np.arange(-ny // 2, ny // 2)
        x = np.arange(-nx // 2, nx // 2)

    yy, xx = np.meshgrid(y, x)
    xdata = np.stack((xx.flatten(), yy.flatten()), axis=1)
    bounds_min = [0.0, x[0], y[0], 0.0, 0.0, -np.pi / 4, 0.0]
    bounds_max = [data_2D.sum(), x[-1], y[-1], x[-1] * 2, y[-1] * 2, np.pi / 4, data_2D.max()]
    popt, _ = curve_fit(generalized_2D_gaussian,
                        xdata,
                        data_2D.flatten(),
                        bounds=[bounds_min, bounds_max])
    amplitude, center_x, center_y, sigma_x, sigma_y, theta, offset = popt

    gaussian_fit = {"amplitude": amplitude,
                    "center_x": center_x,
                    "center_y": center_y,
                    "sigma_x": sigma_x,
                    "fwhm_x": 2.355 * sigma_x,
                    "sigma_y": sigma_y,
                    "fwhm_y": 2.355 * sigma_y,
                    "theta": theta,
                    "offset": offset}

    return gaussian_fit

def differential_evolution_2D_gaussian_fit(data_2D: np.ndarray, x: np.ndarray = None, y: np.ndarray = None) -> dict:
    r"""Fit a 2-d gaussian to the probe intensities (not amplitudes) and return the fit parameters.

    The returned dictionary contains the following fit parameters (as described in [1]_):
        * ``amplitude`` : Amplitude of the fitted gaussian.
        * ``center_x`` : X-offset of the center of the fitted gaussian.
        * ``center_y`` : Y-offset of the center of the fitted gaussian.
        * ``theta`` : Clockwise rotation angle for the gaussian fit. Prior to the rotation, primary axes of the \
          | gaussian  are aligned to the X and Y axes.
        * ``sigma_x`` : Spread (standard deviation) of the gaussian fit along the x-axis (prior to rotation).
        * ``fwhm_x`` : FWHM along x-axis (prior to rotation).
        * ``sigma_y`` : Spread of the gaussian fit along the y-axis (prior to rotation).
        * ``fwhm_y`` : FWHM along y-axis (before rotation).
        * ``offset`` : Constant level of offset applied to the fit.

    Returns
    -------
    out : dict
        Dictionary containing the fit parameters.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    """

    def squared_loss(parameters):
        val = generalized_2D_gaussian(xdata, *parameters)

        return np.sum((data_2D.flatten() - val)**2)


    if x is None and y is None:
        nx = data_2D.shape[-1]
        ny = data_2D.shape[-2]
        y = np.arange(-ny // 2, ny // 2)
        x = np.arange(-ny // 2, nx // 2)

    yy, xx = np.meshgrid(y, x)
    xdata = np.stack((xx.flatten(), yy.flatten()), axis=1)
    bounds_min = [0, x[0], y[0], 0, 0, -np.pi / 4, 0]
    bounds_max = [data_2D.sum(), x[-1], y[-1], x[-1] * 2, y[-1] * 2, np.pi / 4, data_2D.max()]
    bounds_min = np.array(bounds_min) + 1e-7

    result = differential_evolution(generalized_2D_gaussian,
                        xdata,
                        data_2D.flatten(),
                        bounds=[bounds_min, bounds_max])
    amplitude, center_x, center_y, sigma_x, sigma_y, theta, offset = result

    gaussian_fit = {"amplitude": amplitude,
                    "center_x": center_x,
                    "center_y": center_y,
                    "sigma_x": sigma_x,
                    "fwhm_x": 2.355 * sigma_x,
                    "sigma_y": sigma_y,
                    "fwhm_y": 2.355 * sigma_y,
                    "theta": theta,
                    "offset": offset}

    return gaussian_fit
