import numpy as np
from molSimplify.ml.kernels import Masking, PermutationalKernel
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


def test_permutational_kernel():
    permutations = [
        (0, 1),
        (1, 0),
    ]

    rng = np.random.default_rng(0)
    X = rng.normal(1.2, 0.4, size=(10, 2, 4))
    Y = rng.normal(1.1, 0.5, size=(5, 2, 4))

    kernel = PermutationalKernel(shape=(2, 4), permutations=permutations, kernel=RBF(length_scale=1.5))

    # X only
    np.testing.assert_allclose(kernel(X.reshape(10, 8)), kernel(X[:, [1, 0]].reshape(10, 8)))
    # permute X
    np.testing.assert_allclose(kernel(X.reshape(10, 8), Y.reshape(5, 8)),
                               kernel(X[:, [1, 0]].reshape(10, 8), Y.reshape(5, 8)))
    # permute Y
    np.testing.assert_allclose(kernel(X.reshape(10, 8), Y.reshape(5, 8)),
                               kernel(X.reshape(10, 8), Y[:, [1, 0]].reshape(5, 8)))
    # permute both
    np.testing.assert_allclose(kernel(X.reshape(10, 8), Y.reshape(5, 8)),
                               kernel(X[:, [1, 0]].reshape(10, 8), Y[:, [1, 0]].reshape(5, 8)))


def test_permutational_kernel_gradient():
    permutations = [
        (0, 1),
        (1, 0),
    ]

    rng = np.random.default_rng(0)
    X = rng.normal(1.2, 0.4, size=(10, 2, 4))

    l0 = 1.5
    kernel = PermutationalKernel(shape=(2, 4), permutations=permutations, kernel=RBF(length_scale=l0))
    _, K_grad = kernel(X, eval_gradient=True)

    # Compute the gradient numerically
    delta_l = 1e-4
    K_plus = PermutationalKernel(shape=(2, 4), permutations=permutations, kernel=RBF(length_scale=l0 + 0.5 * delta_l))(X)
    K_minus = PermutationalKernel(shape=(2, 4), permutations=permutations, kernel=RBF(length_scale=l0 - 0.5 * delta_l))(X)

    # Since we are looking for the gradient with respect to log(theta), we need to apply the chain
    # rule and divide by the derivative of the log function, i.e., multiply by l0
    K_grad_num = (K_plus - K_minus) / delta_l * l0
    np.testing.assert_allclose(K_grad_num, K_grad.squeeze(), atol=1e-4)
