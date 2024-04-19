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
        return "Masking({0})".format(self.kernel)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.kernel.is_stationary()
