import numpy as np

def get_dales_law_signs(n_dofs: int, frac_positive: float, balanced: bool = True) -> np.ndarray:
    """
    :param n_dofs: the dimension of the square matrix returned.
    :param frac_positive: the fraction of columns with positive values.
        1 - ``frac_positive`` is the fraction of columns with negative values.
    :param balanced: if this flag is provided, we change the average of the
        negative and positive signs so that the total average is 0.
    """
    col_signs = np.array([*([1.] * round(n_dofs * frac_positive)),
                          *([-1.] * round(n_dofs * (1 - frac_positive)))]).reshape((1, n_dofs))

    if (balanced):
        col_signs[col_signs == -1.] *= frac_positive / (1. - frac_positive)

    np.random.shuffle(col_signs)

    return np.ones((n_dofs, 1.)) @ col_signs
