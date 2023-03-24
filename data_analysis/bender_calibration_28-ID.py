import os, glob

import numpy

from single_shot_data import extract_shape_from_measurement_file


bimorph_calibration = numpy.zeros((19, 2)) # voltages X R

bimorph_calibration[0 , 0] = 0.0
bimorph_calibration[1 , 0] = 50.0
bimorph_calibration[2 , 0] = 100.0
bimorph_calibration[3 , 0] = 150.0
bimorph_calibration[4 , 0] = 200.0
bimorph_calibration[5 , 0] = 250.0
bimorph_calibration[6 , 0] = 300.0
bimorph_calibration[7 , 0] = 350.0
bimorph_calibration[8 , 0] = 400.0
bimorph_calibration[9 , 0] = 450.0
bimorph_calibration[10, 0] = 500.0
bimorph_calibration[11, 0] = 550.0
bimorph_calibration[12, 0] = 600.0
bimorph_calibration[13, 0] = 650.0
bimorph_calibration[14, 0] = 700.0
bimorph_calibration[15, 0] = 750.0
bimorph_calibration[16, 0] = 800.0
bimorph_calibration[17, 0] = 850.0
bimorph_calibration[18, 0] = 900.0

bimorph_calibration[0 , 1] = 1.835
bimorph_calibration[1 , 1] = 1.798
bimorph_calibration[2 , 1] = 1.729
bimorph_calibration[3 , 1] = 1.652
bimorph_calibration[4 , 1] = 1.527
bimorph_calibration[5 , 1] = 1.417
bimorph_calibration[6 , 1] = 1.197
bimorph_calibration[7 , 1] = 0.991
bimorph_calibration[8 , 1] = 0.571
bimorph_calibration[9 , 1] = numpy.nan
bimorph_calibration[10, 1] = numpy.nan
bimorph_calibration[11, 1] = -6.606
bimorph_calibration[12, 1] = 36.322
bimorph_calibration[13, 1] = 8.291
bimorph_calibration[14, 1] = 5.624
bimorph_calibration[15, 1] = 4.609
bimorph_calibration[16, 1] = 4.073
bimorph_calibration[17, 1] = 3.762
bimorph_calibration[18, 1] = 3.54

bender_calibration = numpy.zeros((5, 3))
bender_calibration[0, 0] = -150.0
bender_calibration[1, 0] = -125.0
bender_calibration[2, 0] = -100.0
bender_calibration[3, 0] = -75.0
bender_calibration[4, 0] = -50.0

bender_calibration[0, 1] = 0.811
bender_calibration[1, 1] = 1.184
bender_calibration[2, 1] = 1.518
bender_calibration[3, 1] = 1.776
bender_calibration[4, 1] = 1.975

bender_calibration[0, 2] = 0.913
bender_calibration[1, 2] = 1.341
bender_calibration[2, 2] = 1.631
bender_calibration[3, 2] = 1.864
bender_calibration[4, 2] = 2.040

distance = [2.5, 3.63]

from matplotlib import pyplot as plt

def process_calibration(calibration, index, distance):
    data = numpy.zeros((calibration.shape[0], 2))

    data[:, 0] = calibration[: , 0]
    data[:, 1] = 1e-3/(distance - calibration[:, index])

    return data


def plot_axis(ax, data, title, degree, xlim, ylim, color='blue', excluded=[], top=True):
    ax.plot(data[:, 0], data[:, 1], "o", color=color)
    text = "Fit parameters:\n"
    xx = numpy.ma.array(data[:, 0], mask=False)
    yy = numpy.ma.array(data[:, 1], mask=False)

    if len(excluded) > 0:
        xx.mask[excluded] = True
        yy.mask[excluded] = True

    xx = xx.compressed()
    yy = yy.compressed()
    param = numpy.polyfit(x=xx, y=yy, deg=degree)
    for i in range(len(param)): text += "p" + str(i) + "=%0.10f" % (param[i],) + "\n"
    text += "\n"

    print(title + " - Fit parameters:\n" + text)

    ax.plot(xx, numpy.poly1d(param)(xx), "-.", lw=0.5, color=color)
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("$ \\frac{1}{Q}$ [$mm^{-1}$]")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if top:
        ax.text(x=ax.get_xlim()[0] + (ax.get_xlim()[1]-ax.get_xlim()[0])*0.05,
                y=ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.7,
                s=text, fontsize=12)
    else:
        ax.text(x=ax.get_xlim()[0] + (ax.get_xlim()[1]-ax.get_xlim()[0])*0.05,
                y=ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05,
                s=text, fontsize=12)

    ax.set_title(title)

    return param

vertical = process_calibration(bimorph_calibration, 1, distance[0])
horizontal_upward = process_calibration(bender_calibration, 2, distance[1])
horizontal_downward = process_calibration(bender_calibration, 1, distance[1])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

fig.suptitle("Benders Calibration")

param_vertical = plot_axis(axes[0, 0], vertical,            "Vertical",             1, xlim=[-50, 950], ylim=[-0.9e-3, 1.6e-3], color='red', top=False, excluded=[9, 10])
axes[1, 0].remove()
param_horizontal_u = plot_axis(axes[0, 1], horizontal_upward,   "Horizontal: upward",   1, xlim=[-160, -40], ylim=[0.32e-3, 0.65e-3])
param_horizontal_d = plot_axis(axes[1, 1], horizontal_downward, "Horizontal: downward", 1, xlim=[-160, -40], ylim=[0.32e-3, 0.65e-3])

fig.tight_layout()

print("384 -> ",  str(1/(param_vertical[0]*384 + param_vertical[1])))
print("-170 -> ", str(1/(param_horizontal_u[0]*-170 + param_horizontal_u[1])))
print("-157 -> ", str(1/(param_horizontal_d[0]*-155 + param_horizontal_d[1])))

print("2290 -> ",  str((1/2290 - param_vertical[1]) / param_vertical[0]))
print("3095 -> ",  str((1/3095 - param_horizontal_u[1]) / param_horizontal_u[0]))
print("3095 -> ",  str((1/3095 - param_horizontal_d[1]) / param_horizontal_d[0]))



plt.show()


