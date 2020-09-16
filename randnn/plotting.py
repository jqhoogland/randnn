"""

Contains basic functions for plotting various elements of this project:
- (low-dimensional projections of) trajectories and their averages
- Lyapunov spectra
- etc.

Author: Jesse Hoogland
Year: 2020

"""

import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory_avg(trajectory):
    avg_trajectory = np.mean(trajectory, axis=1)
    stdev_trajectory = np.std(trajectory, axis=1)
    plt.errorbar(np.arange(avg_trajectory.size),
                 avg_trajectory,
                 yerr=stdev_trajectory)
    plt.show()
