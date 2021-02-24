import numpy as np

def get_dales_law_signs(n_dofs: int, frac_positive: float) -> np.ndarray:
    """
    :param n_dofs: the dimension of the square matrix returned.
    :param frac_positive: the fraction of columns with positive values.
        1 - ``frac_positive`` is the fraction of columns with negative values.
    """
    col_signs = np.array([*([1] * round(n_dofs * frac_positive)),
                          *([-1] * round(n_dofs * ( 1 - frac_positive)))]).reshape((1, n_dofs))

    np.random.shuffle(col_signs)

    return np.ones((n_dofs, 1)) @ col_signs



