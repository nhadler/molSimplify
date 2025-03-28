import numpy as np
from sklearn.gaussian_process.kernels import Kernel


class Masking(Kernel):
    def __init__(self, mask, kernel):
        self.mask = mask
        self.kernel = kernel

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        self.kernel.theta = theta

    @property
    def bounds(self):
        return self.kernel.bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            return self.kernel(X[:, self.mask], Y=None, eval_gradient=eval_gradient)
        return self.kernel(
            X[:, self.mask], Y=Y[:, self.mask], eval_gradient=eval_gradient
        )

    def diag(self, X):
        return self.kernel.diag(X[:, self.mask])

    def __repr__(self):
        return f"Masking({self.kernel})"

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.kernel.is_stationary()


class PermutationalKernel(Kernel):
    def __init__(self, shape, permutations, kernel):
        self.shape = shape
        self.permutations = permutations
        self.kernel = kernel

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        self.kernel.theta = theta

    @property
    def bounds(self):
        return self.kernel.bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        n = X.shape[0]
        n_perms = len(self.permutations)

        # The main idea of this implementation is to vectorize the double loop over the
        # permutations. This is done by building a new X array that includes all possible
        # permutations of the input features. The kernel is then evaluated on this reshaped
        # array and the result is averaged over the permutations.
        X_reshaped = X.reshape(-1, *self.shape)
        X_permuted = np.stack(
                [X_reshaped[:, perm].reshape(X.shape) for perm in self.permutations],
                axis=1
            ).reshape(n*n_perms, -1)

        if eval_gradient:
            if Y is not None:
                raise ValueError("Gradient can only be evaluated when Y is None.")

            K, K_grad = self.kernel(X_permuted, eval_gradient=True)
            # Reshape and average over the permutations
            return (
                K.reshape(n, n_perms, n, n_perms).sum(axis=(1, 3)) / n_perms**2,
                K_grad.reshape(n, n_perms, n, n_perms, -1).sum(axis=(1, 3))
                / n_perms**2,
            )

        if Y is None:
            # Reshape and average over the permutations
            return self.kernel(X_permuted).reshape(n, n_perms, n, n_perms).sum(axis=(1, 3)) / n_perms ** 2

        m = Y.shape[0]
        Y_reshaped = Y.reshape(-1, *self.shape)
        Y_permuted = np.stack(
                [Y_reshaped[:, perm].reshape(Y.shape) for perm in self.permutations],
                axis=1
            ).reshape(m*n_perms, -1)
        # Reshape and average over the permutations
        return self.kernel(X_permuted, Y_permuted).reshape(n, n_perms, m, n_perms).sum(axis=(1, 3)) / n_perms ** 2

    def diag(self, X):
        # TODO: More efficient implementation
        return np.diag(self(X))

    def __repr__(self):
        return f"PermutationalKernel(shape={self.shape}, permutations={self.permutations}, kernel={self.kernel})"

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.kernel.is_stationary()
