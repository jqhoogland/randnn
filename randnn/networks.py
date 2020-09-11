"""

Contains the various networks I'll be exploring,
i.e. the pairs of neuronal dynamics and network topologies.

Author: Jesse Hoogland
Year: 2020

"""

class ContinuousNeurons(Trajectory):
    """
    NOTE: I could have used multiple inheritance here (from Trajectory and Topology),
    but that would have quickly become sloppy.

    """
        def __init__(self, n_nodes, coupling, self_interaction=False, init_state=None, step_size=0.001, vectorized=True, n_dofs=3):
            pass
