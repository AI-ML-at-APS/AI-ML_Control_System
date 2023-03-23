import os, glob

import numpy

from single_shot_data import extract_shape_from_measurement_file

root_dir = "/Users/lrebuffi/Library/CloudStorage/Box-Box/Luca_Documents"
if not os.path.exists(root_dir): root_dir = "/Users/lrebuffi/Box Sync/Luca_Documents"

home_dir = os.path.join(root_dir, "AI-ML/34-ID-C/Beamtime/Jul14-2022/bender_calibration")

directories = {"horizontal" : [os.path.join(home_dir, "slits_full", "horizontal"), os.path.join(home_dir, "slits_150_150", "horizontal")],
               "vertical"   : [os.path.join(home_dir, "slits_full", "vertical"),   os.path.join(home_dir, "slits_150_150", "vertical")]}

dataset   = {"horizontal" : "line_curve_x", "vertical" : "line_curve_y"}
distances =  {"horizontal" : 387, "vertical" : 493}

from matplotlib import pyplot as plt

def process_direction(direction):
    directory_full = directories[direction][0]
    directory_150  = directories[direction][1]

    def process_directory(directory, direction):
        image_files_list = sorted(
            glob.glob(os.path.join(os.path.join(directory, "image_*"), 'single_*.hdf5')),
            key=lambda x: float(os.path.dirname(x).split(os.sep)[-1].split('_')[1][1:6]))

        upward_data   = numpy.zeros((len(image_files_list), 2))
        downward_data = numpy.zeros((len(image_files_list), 2))

        for index in range(len(image_files_list)):
            image_file = image_files_list[index]

            radius, _, x, _ = extract_shape_from_measurement_file(image_file, dataset[direction], -1, 20)

            tmp = image_file.split(sep="image_")[1]

            half   = int(len(radius)/2)
            q_up   = distances[direction] - 1000/numpy.average(radius[0:half])
            q_down = distances[direction] - 1000/numpy.average(radius[half:])

            upward_data[index, 0] = float(tmp[1:6])
            upward_data[index, 1] = 1/q_up

            downward_data[index, 0] = float(tmp[8:13])
            downward_data[index, 1] = 1/q_down

        return upward_data, downward_data

    upward_data_full, downward_data_full = process_directory(directory_full, direction)
    upward_data_150, downward_data_150 = process_directory(directory_150, direction)

    return [upward_data_full, upward_data_150], [downward_data_full, downward_data_150]

def plot_axis(ax, data, title, degree, xlim, ylim, color='blue', escluded=[]):
    ax.plot(data[:, 0], data[:, 1], "bo", color=color)
    text = "Fit parameters:\n"
    xx = numpy.ma.array(data[:, 0], mask=False)
    yy = numpy.ma.array(data[:, 1], mask=False)

    if len(escluded) > 0:
        xx.mask[escluded] = True
        yy.mask[escluded] = True

    xx = xx.compressed()
    yy = yy.compressed()
    param = numpy.polyfit(x=xx, y=yy, deg=degree)
    for i in range(len(param)): text += "p" + str(i) + "=%0.7f" % (param[i],) + "\n"
    text += "\n"
    ax.plot(xx, numpy.poly1d(param)(xx), "-.", lw=0.5, color=color)
    ax.set_xlabel("Motor Position [mm]")
    ax.set_ylabel("$ \\frac{1}{Q}$ [$mm^{-1}$]")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.text(x=ax.get_xlim()[0] + (ax.get_xlim()[1]-ax.get_xlim()[0])*0.05,
            y=ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.7,
            s=text, fontsize=12)
    ax.set_title(title)

horizontal_upward, horizontal_downward = process_direction(direction="horizontal")
vertical_upward, vertical_downward     = process_direction(direction="vertical")

both = False

if both:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    fig.suptitle("Bender Calibration")

    plot_axis(axes[0, 0], horizontal_upward[0],   "Horizontal, slits full: upward",   1, xlim=[200, 230], ylim= [7.7e-3, 8.6e-3])
    plot_axis(axes[0, 1], horizontal_downward[0], "Horizontal, slits full: downward", 1, xlim=[95, 125],  ylim= [7.7e-3, 8.6e-3])
    plot_axis(axes[1, 0], horizontal_upward[1],   "Horizontal, slits 150: upward",    1, xlim=[200, 230], ylim= [7.7e-3, 8.6e-3])
    plot_axis(axes[1, 1], horizontal_downward[1], "Horizontal, slits 150: downward",  1, xlim=[95, 125],  ylim= [7.7e-3, 8.6e-3])
    plot_axis(axes[0, 2], vertical_upward[0],     "Vertical, slits full: upward",     1, xlim=[125, 150], ylim= [4.15e-3, 4.45e-3])
    plot_axis(axes[0, 3], vertical_downward[0],   "Vertical, slits full: downward",   1, xlim=[230, 255], ylim= [4.15e-3, 4.45e-3])
    plot_axis(axes[1, 2], vertical_upward[1],     "Vertical, slits 150: upward",      1, xlim=[125, 150], ylim= [4.15e-3, 4.45e-3])
    plot_axis(axes[1, 3], vertical_downward[1],   "Vertical, slits 150: downward",    1, xlim=[230, 255], ylim= [4.15e-3, 4.45e-3])
else:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_axis(axes[0, 0], horizontal_upward[1],   "Horizontal: upward",   1, xlim=[204, 227], ylim=[7.7e-3, 8.5e-3])
    plot_axis(axes[1, 0], horizontal_downward[1], "Horizontal: downward", 1, xlim=[99, 122],  ylim=[7.7e-3, 8.5e-3])
    plot_axis(axes[0, 1], vertical_upward[1],     "Vertical: upward",     1, xlim=[126, 150], ylim=[4.17e-3, 4.45e-3], color='red')
    plot_axis(axes[1, 1], vertical_downward[1],   "Vertical: downward",   1, xlim=[232, 255], ylim=[4.17e-3, 4.45e-3], color='red')

    fig.tight_layout()

'''
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Bender Calibration")

directories = {"horizontal" : [os.path.join(home_dir, "slits_full", "horizontal_2"), os.path.join(home_dir, "slits_150_150", "horizontal_2")],
               "vertical"   : [os.path.join(home_dir, "slits_full", "vertical_2"),   os.path.join(home_dir, "slits_150_150", "vertical_2")]}

horizontal_upward, horizontal_downward = process_direction(direction="horizontal")
vertical_upward, vertical_downward     = process_direction(direction="vertical")

plot_axis(axes[0, 0], horizontal_upward[0],   "Horizontal, slits full: upward",   2, 1/115, escluded=[0])
plot_axis(axes[0, 1], horizontal_downward[0], "Horizontal, slits full: downward", 2, 1/115, escluded=[0, 1, 10])
plot_axis(axes[1, 0], horizontal_upward[1],   "Horizontal, slits 150: upward",    2, 1/115, escluded=[])
plot_axis(axes[1, 1], horizontal_downward[1], "Horizontal, slits 150: downward",  2, 1/115, escluded=[])
plot_axis(axes[0, 2], vertical_upward[0],     "Vertical, slits full: upward",     2, 1/220, escluded=[])
plot_axis(axes[0, 3], vertical_downward[0],   "Vertical, slits full: downward",   2, 1/220, escluded=[0, 1, 2])
plot_axis(axes[1, 2], vertical_upward[1],     "Vertical, slits 150: upward",      2, 1/220, escluded=[0, 1])
plot_axis(axes[1, 3], vertical_downward[1],   "Vertical, slits 150: downward",    2, 1/220, escluded=[0, 1])
'''
plt.show()


