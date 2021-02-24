import numpy as np

from randnn.signs import get_dales_law_signs

def test_dales_law():
    for n in range(10, 1000, 200):
        for r in np.arange(0., 1., 0.1):
            signs = get_dales_law_signs(n, r)

            col_signs = np.sum(signs, axis=0) / n
            print(signs, col_signs)
            assert np.all(col_signs == col_signs[np.logical_or(col_signs == 1., col_signs == -1.)]), "Same value across col"
            assert np.allclose(np.sum(col_signs[col_signs == 1]), round(n*r)), "Fraction that are positive"