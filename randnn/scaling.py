"""

Contains methods to perform scaling analysis (i.e. with system size or with some other adjustable parameter)

Author: Jesse Hoogland
Year: 2020

"""
import numpy as np
from typing import Callable, List, Any, Optional

from pynamics.trajectories import Trajectory
from .networks import ContinuousNN



def scaling_analysis(scaling_kwargs: List[dict],
                     init_fn: Callable[[Any], Any],
                     analysis_fns=List[Callable[[Any], np.float64]],
                     init_kwargs: Optional[dict] = {}) -> np.ndarray:
    """
    :param scaling_vars: dict of keyword argument/value pairs to provide ContinuousNN when initializing
    :param init_fn: a function which takes as key-word arguments scaling_vars and init_kwargs
    :param analysis_fns: callable which returns a numerical value from the Continuous NN state.
    :param
    :returns results: np.ndarray of shape (n_scaling_vars, 1 + n_scaling_fns),
        where each row corresponds to the results for one scaling variable:
        the first column is the value of the scaling variable and the remaining entries
        are the results of the scaling function (in the same order)
    """

    results = []
    for scaling_kwarg in scaling_kwargs:
        _results = []
        logging.info("Scaling var: %s", scaling_kwarg)
        result = init_fn(**scaling_kwarg, **init_kwargs)
        for fn in analysis_fns:
            _results.append(fn(result))

        results.append(_results)

    scaling_vars = list(
        map(lambda scaling_kwarg: (list(scaling_kwarg.values()))[0],
            scaling_kwargs))
    return np.concatenate([np.array([scaling_vars]).T, results], axis=1)


def scaling_n_dofs():
    def init_fn(n_steps=1000, n_burn_in=200, t_ons=10, **kwargs):
        system = ContinuousNN(**kwargs)
        trajectory = system.run(n_steps=n_steps, n_burn_in=n_burn_in)

        return system.get_lyapunov_spectrum(trajectory, t_ons=t_ons)

    analysis_fns = [get_attractor_dimension]
    scaling_kwargs = [{
        "n_dofs": 10
    }, {
        "n_dofs": 100
    }, {
        "n_dofs": 500
    }, {
        "n_dofs": 1000
    }, {
        "n_dofs": 2000
    }, {
        "n_dofs": 5000
    }]

    init_kwargs = {"coupling_strength": COUPLING_STRENGTH, "t_ons": 200}

    return scaling_analysis(scaling_kwargs, init_fn, analysis_fns, init_kwargs)
