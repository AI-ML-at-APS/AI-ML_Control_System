import numpy as np

def generalized2dGaussian(xdata_tuple: np.ndarray, # array consisting of 2d points.
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

def calculate_gaussian_fit(data_2d) -> dict:
    r"""Fit a 2-d gaussian to the probe intensities (not amplitudes) and return the fit parameters.

    The returned dictionary contains the following fit parameters (as described in [1]_):
        * ``amplitude`` : Amplitude of the fitted gaussian.
        * ``center_x`` : X-offset of the center of the fitted gaussian.
        * ``center_y`` : Y-offset of the center of the fitted gaussian.
        * | ``theta`` : Clockwise rotation angle for the gaussian fit. Prior to the rotation, primary axes of the \
          | gaussian  are aligned to the X and Y axes.
        * ``sigma_x`` : Spread (standard deviation) of the gaussian fit along the x-axis (prior to rotation).
        * ``fwhm_x`` : FWHM along x-axis (prior to rotation).
        * ``sigma_y`` : Spread of the gaussian fit along the y-axis (prior to rotation).
        * ``fwhm_y`` : FWHM along y-axis (before rotation).
        * | ``offset`` : Constant level of offset applied to the intensity throughout the probe array. This could,
          | for instance, represent the level of background radiation.

    Returns
    -------
    out : dict
        Dictionary containing the fit parameters.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    """
    from scipy.optimize import curve_fit

    nx = data_2d.shape[-1]
    ny = data_2d.shape[-2]
    # intensities = np.fft.ifftshift(np.abs(self.wavefront)**2)
    y = np.arange(-ny // 2, ny // 2)
    x = np.arange(-ny // 2, nx // 2)
    yy, xx = np.meshgrid(y, x)
    xdata = np.stack((xx.flatten(), yy.flatten()), axis=1)
    bounds_min = [0, x[0], y[0], 0, 0, -np.pi / 4, 0]
    bounds_max = [data_2d.sum(), x[-1], y[-1], x[-1] * 2, y[-1] * 2, np.pi / 4, data_2d.max()]
    popt, _ = curve_fit(generalized2dGaussian,
                        xdata,
                        data_2d.flatten(),
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