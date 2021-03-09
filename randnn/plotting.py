import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


def plot_coupling_matrix_spectrum(matrix, radius=None):
    N, _ = matrix.shape  # Assuming square
    if radius is None:
        radius = np.std(matrix) * np.sqrt(N)

    C = np.zeros((N, N), dtype=np.complex_)
    eigs, _ = np.linalg.eig(matrix + C)

    fix, ax = plt.subplots()
    ax.scatter(eigs.real, eigs.imag, color=(0, 0, 0), s=2, alpha=0.5)

    plt.axis("square")

    if radius is not None:
        circle = patches.Circle((0, 0),
                                radius,
                                fill=False,
                                color="k",
                                linewidth=2,
                                edgecolor='b')
        ax.add_patch(circle)

    ax.axvline(x=1., linestyle="--")

    ax.set_title("Eigenvalue spectrum of the coupling matrix $J_{ij}$")
    ax.set_ylabel("Complex part")
    ax.set_xlabel("Real part")
