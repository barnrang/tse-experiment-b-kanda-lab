from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime

from absl import app
from absl import flags

import numpy as np
import matplotlib.pyplot as plt
import pandas

FLAGS = flags.FLAGS

flags.DEFINE_string('filepath', 'data/2018_data.xlsx', 'Path to data file')


def parse_time(time_str: str) -> float:
    """ Parse a time string to unix time (seconds since the Epoch)

    Example: 12-06-08 15:39:37.8s -> ???
    """
    return datetime.strptime(time_str, "%y-%m-%d %H:%M:%S.%fs").timestamp()


def rot_matrix(vec):
    """ Calculate a rotation matrix A such that (A * vec) has only
    the first component non-zero
    """

    # Angle between Ox and proj(vec, O-xy)
    alpha = np.arctan2(vec[1], vec[0])
    # Angle between O-xy and vec
    beta = np.arctan2(vec[2], np.sqrt(vec[0]**2 + vec[1]**2))

    rot_alpha = np.array([
        [np.cos(alpha), np.sin(alpha), 0],
        [-np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1],
    ])
    rot_beta = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)],
    ])
    return np.dot(rot_beta, rot_alpha)


def plot(u_dev, v_dev, w_dev, temp_dev, time):
    """ Plot graphs
    """

    fig = plt.figure(figsize=(10, 17))
    plot_count = 6

    plots = [
        (u_dev, "u'"),
        (v_dev, "v'"),
        (w_dev, "w'"),
        (temp_dev, "T'"),
        (u_dev * w_dev, "u'w'"),
        (w_dev * temp_dev, "w'T'"),
    ]

    # Draw subplots
    all_axes = []
    for index, (y_data, y_label) in enumerate(plots):
        ax = fig.add_subplot(plot_count, 1, index + 1)  # pylint: disable=invalid-name
        ax.plot(time, y_data)
        ax.set_xlim(time[0], time[-1])
        ax.set_ylabel(y_label, rotation=0, labelpad=20)
        ax.grid()
        all_axes.append(ax)

    # Sync all x-axis together
    all_axes[0].get_shared_x_axes().join(*all_axes)

    fig.align_ylabels()
    fig.set_tight_layout(True)
    return fig


def read_data(filename):
    """ Read data from an xlsx file

    Returns:
        time: seconds from the start of measurement
        v_wind: [3 x N] matrix containing wind speed
        temp: Temperature
    """

    data = pandas.read_excel(filename, header=[0], index_col=0, usecols=4)

    time = np.array([parse_time(time_str) for time_str in data.index.values])
    time -= time[0]

    v_wind = data.iloc[:, :3].values.T
    temp = data.iloc[:, 3].values

    return time, v_wind, temp


def main(argv):
    del argv  # Unused

    time, v_wind, temp = read_data(FLAGS.filepath)

    v_wind_mean = np.mean(v_wind, axis=1)

    # Normalized data
    v_wind_norm = np.dot(rot_matrix(v_wind_mean), v_wind)

    # In the normalized wind speed, means of 2nd and 3rd component must be zero
    assert np.all(np.mean(v_wind_norm, axis=1)[1:] < 1e-9)

    # Deviation from mean of wind speed in streamwise, spanwise and ground-normal
    # direction
    u_dev, v_dev, w_dev = v_wind_norm - np.mean(
        v_wind_norm, axis=1, keepdims=True)

    # Deviation from mean of temperature
    temp_dev = temp - np.mean(temp)

    plot(u_dev, v_dev, w_dev, temp_dev, time)
    plt.show()


if __name__ == '__main__':
    app.run(main)
