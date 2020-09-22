"""

Contains methods to perform scaling analysis (i.e. with system size or with some other adjustable parameter)

Author: Jesse Hoogland
Year: 2020

"""
from typing import Callable, List, Any
from .trajectories import Trajectory


def scaling_analysis(init_fn: Callable[[Union[float, np.ndarray]],
                                       Trajectory], scaling_vars: list,
                     analysis_fns: List[Callable[[np.ndarray], np.ndarray]],
                     **kwargs) -> list:

    results = []
    for scaling_var in scaling_vars:
        _results = []
        system = init_fn(scaling_var)
        trajectory = system.run(**kwargs)

        for fn in analysis_fns:
            _results.push(fn(trajectory))

        results.push((scaling_var, _results))

    return results
