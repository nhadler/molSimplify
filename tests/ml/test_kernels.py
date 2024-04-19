import numpy as np
from molSimplify.ml.kernels import Masking
from sklearn.gaussian_process.kernels import RBF


def test_masking():
    rng = np.random.default_rng(0)
    X = rng.normal(1.2, 0.4, size=(10, 4))
    Y = rng.normal(1.1, 0.5, size=(7, 4))

    mask = [True, False, True, False]
    kernel = Masking(mask, RBF(length_scale=1.5))
    kernel_ref = RBF(length_scale=1.5)

    np.testing.assert_allclose(kernel(X), kernel_ref(X[:, mask]))
    np.testing.assert_allclose(kernel(X, Y), kernel_ref(X[:, mask], Y[:, mask]))

    # Second use case, with a slice mask, equivalent to ::2
    kernel2 = Masking(slice(None, None, 2), RBF(length_scale=1.5))
    np.testing.assert_allclose(kernel(X, Y), kernel2(X, Y))
