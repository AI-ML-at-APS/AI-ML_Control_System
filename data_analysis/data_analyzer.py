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
            upward_data[index, 1] = q_up

            downward_data[index, 0] = float(tmp[8:13])
            downward_data[index, 1] = q_down

        return upward_data, downward_data

    upward_data_full, downward_data_full = process_directory(directory_full, direction)
    upward_data_150, downward_data_150 = process_directory(directory_150, direction)

    return [upward_data_full, upward_data_150], [downward_data_full, downward_data_150]

def plot_axis(ax, data, title, degree, ylim, escluded=[]):
    ax.plot(data[:, 0], data[:, 1], "bo")
    text = ""
    xx = numpy.ma.array(data[:, 0], mask=False)
    yy = numpy.ma.array(data[:, 1], mask=False)

    if len(escluded) > 0:
        xx.mask[escluded] = True
        yy.mask[escluded] = True

    xx = xx.compressed()
    yy = yy.compressed()
    param = numpy.polyfit(x=xx, y=yy, deg=degree)
    for i in range(len(param)): text += "p" + str(i) + "=" +str(round(param[i], 6)) + "\n"
    text += "\n"
    ax.plot(xx, numpy.poly1d(param)(xx))

    ax.text(x=ax.get_xlim()[0] * 1.001, y=ylim, s=text, fontsize=8)
    ax.set_title(title)

horizontal_upward, horizontal_downward = process_direction(direction="horizontal")
vertical_upward, vertical_downward     = process_direction(direction="vertical")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Bender Calibration")

plot_axis(axes[0, 0], horizontal_upward[0],   "Horizontal, slits full: upward",   1, 120)
plot_axis(axes[0, 1], horizontal_downward[0], "Horizontal, slits full: downward", 1, 120)
plot_axis(axes[1, 0], horizontal_upward[1],   "Horizontal, slits 150: upward",    1, 120)
plot_axis(axes[1, 1], horizontal_downward[1], "Horizontal, slits 150: downward",  1, 120)
plot_axis(axes[0, 2], vertical_upward[0],     "Vertical, slits full: upward",     1, 228)
plot_axis(axes[0, 3], vertical_downward[0],   "Vertical, slits full: downward",   1, 228)
plot_axis(axes[1, 2], vertical_upward[1],     "Vertical, slits 150: upward",      1, 228)
plot_axis(axes[1, 3], vertical_downward[1],   "Vertical, slits 150: downward",    1, 228)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Bender Calibration")

directories = {"horizontal" : [os.path.join(home_dir, "slits_full", "horizontal_2"), os.path.join(home_dir, "slits_150_150", "horizontal_2")],
               "vertical"   : [os.path.join(home_dir, "slits_full", "vertical_2"),   os.path.join(home_dir, "slits_150_150", "vertical_2")]}

horizontal_upward, horizontal_downward = process_direction(direction="horizontal")
vertical_upward, vertical_downward     = process_direction(direction="vertical")

plot_axis(axes[0, 0], horizontal_upward[0],   "Horizontal, slits full: upward",   2, 115, escluded=[0])
plot_axis(axes[0, 1], horizontal_downward[0], "Horizontal, slits full: downward", 2, 115, escluded=[0, 1, 10])
plot_axis(axes[1, 0], horizontal_upward[1],   "Horizontal, slits 150: upward",    2, 115, escluded=[])
plot_axis(axes[1, 1], horizontal_downward[1], "Horizontal, slits 150: downward",  2, 115, escluded=[])
plot_axis(axes[0, 2], vertical_upward[0],     "Vertical, slits full: upward",     2, 220, escluded=[])
plot_axis(axes[0, 3], vertical_downward[0],   "Vertical, slits full: downward",   2, 220, escluded=[0, 1, 2])
plot_axis(axes[1, 2], vertical_upward[1],     "Vertical, slits 150: upward",      2, 220, escluded=[0, 1])
plot_axis(axes[1, 3], vertical_downward[1],   "Vertical, slits 150: downward",    2, 220, escluded=[0, 1])

plt.show()


