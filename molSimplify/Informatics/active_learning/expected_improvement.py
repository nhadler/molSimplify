import numpy as np
from scipy.stats import norm
from typing import Tuple


def get_2D_pareto_indices(points: np.ndarray) -> np.ndarray:
    """Calculate the indices of the points that make up the 2D pareto front.

    Parameters
    ----------
    points : array_like, shape (N,2)
        two dimensional array of all observations

    Returns
    -------
    pareto_indices : ndarray of int, shape (M,)
        list of indices that define the pareto front
    """

    # Follows https://en.wikipedia.org/wiki/Maxima_of_a_point_set#Two_dimensions
    # Modified for minimization of both dimensions, changed are HIGHLIGHTED by
    # upper case text.
    # 1. Sort the points in one of the coordinate dimensions
    indices = np.argsort(points[:, 0])
    pareto_indices = []
    y_min = np.inf
    # 2. For each point, in INCREASING x order, test whether its y-coordinate is
    #    SMALLER than the MINIMUM y-coordinate of any previous point
    for ind in indices:
        if points[ind, 1] < y_min:
            # If it is, save the points as one of the maximal points, and remember
            # its y-coordinate as the SMALLEST seen so far
            pareto_indices.append(ind)
            y_min = points[ind, 1]
    return np.array(pareto_indices)


def get_2D_EI(pred_mean: np.ndarray, pred_std: np.ndarray,
              pareto_points: np.ndarray, method: str = "aug") -> np.ndarray:
    """Calculates the two dimensional expected improvement following equation
    (16) in A. J. Keane, AIAA Journal, 44, 4, 2006, https://doi.org/10.2514/1.16875

    Parameters
    ----------
    pred_mean : array_like, shape (N,2)
        the predicted mean from a ML model for both target properties
    pred_std : array_like, shape (N,2)
        the std mean from a ML model for both target properties
    pareto_points : array_like, shape (M, 2)
        the points on the current Pareto front, must be correctly ordered
    method: str, default = "aug"
        use the probability distribution that augments ("aug") or dominates ("dom")
        the current Pareto front. "mix" is an average of the two and corresponds to
        straight line connections between the points on the Pareto front.
    Returns
    -------
    np.ndarray, shape (N,)
        array of the expected improvement values
    """
    PI, centroid = get_2D_PI_and_centroid(
        pred_mean, pred_std, pareto_points, method=method)

    # Find the closest point on the Pareto front to the centroid. Build a 3d array
    # of shape (N, M, 2) with the coordinate differences, square that, sum over the
    # last dimension, i.e., x^2 + y^2, take the square root and finally find the
    # smallest over the dimension of M Pareto front points
    min_dist = np.min(
        np.sqrt(
            np.sum((centroid[:, np.newaxis, :] - pareto_points[np.newaxis, :, :])**2,
                   axis=2)
                ),
        axis=1)
    # Equation (16)
    return PI * min_dist


def get_2D_PI_and_centroid(pred_mean: np.ndarray, pred_std: np.ndarray,
                           pareto_points: np.ndarray, method: str = "aug"
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the two dimensional probability of improvement following and the
    first moments of that probability distribution according to equations (14) / (16)
    and (18) / (19) in A. J. Keane, AIAA Journal, 44, 4, 2006,
    https://doi.org/10.2514/1.16875

    Note that there is a mistake in the final term in equations (18) and (19) of the
    paper.

    Parameters
    ----------
    pred_mean : array_like, shape (N,2)
        the predicted mean from a ML model for both target properties
    pred_std : array_like, shape (N,2)
        the std mean from a ML model for both target properties
    pareto_points : array_like, shape (M, 2)
        the points on the current Pareto front, must be correctly ordered
    method: str, default = "aug"
        use the probability distribution that augments ("aug") or dominates ("dom")
        the current Pareto front. "mix" is an average of the two and corresponds to
        straight line connections between the points on the Pareto front.
    Returns
    -------
    np.ndarray, shape (N,)
        array of the probability of improvement values
    np.ndarray, shape (N,2)
        array of the centroids
    """
    # Test that the pareto_points make a valid front (Important for the order of the
    # integrals)
    # The x values must be strictly ascending
    if any(np.diff(pareto_points[:, 0]) < 0.0):
        raise ValueError("The x values of the Pareto front must be in ascending order")
    # The corresponding y values must be strictly descending
    if any(np.diff(pareto_points[:, 1]) > 0.0):
        raise ValueError("The y values of the Pareto front must be in descending order")

    # Variables consistent with naming in the paper:
    mu_1 = pred_mean[:, 0]
    s_1 = pred_std[:, 0]
    mu_2 = pred_mean[:, 1]
    s_2 = pred_std[:, 1]
    # Dropping the star in the notation of the Pareto front points for brevity
    f_1e = pareto_points[:, 0]
    f_2e = pareto_points[:, 1]

    # Helper function for the centroid calculations that combines the result of the
    # integration by parts: mu * cdf - s * pdf into one function:
    def int_by_parts(f_e, mu, s):
        return mu * norm.cdf(f_e, mu, s) - s * norm.pdf(f_e, mu, s)

    # NOTE: instead of using the integration variables y_1 and y_2 the comments just
    # refer to the two dimensions as x and y.
    # First part is the integral dx from -inf to the x-coordinate of the left most
    # Pareto point and the integral dy from -inf to inf, later of which is 1
    PI = norm.cdf(f_1e[0], mu_1, s_1)

    # Similarly for the centroid
    centroid = np.zeros_like(pred_mean)
    # Integrate dx by parts and multiply be the integral dy, which again is 1
    centroid[:, 0] = int_by_parts(f_1e[0], mu_1, s_1)
    # Integral over dx (same as PI) multiplied by the integral dy from -inf to inf,
    # Perform integration by parts, where cdf(inf)=1 and pdf(inf)=0,
    # therefore, int y dy from -inf to inf is just mu_2
    centroid[:, 1] = norm.cdf(f_1e[0], mu_1, s_1) * mu_2

    for i in range(len(pareto_points) - 1):
        # First the integral dx from the x-coordinate of Pareto point i to the
        # x-coordinate of Pareto point i+1
        int_dx = norm.cdf(f_1e[i+1], mu_1, s_1) - norm.cdf(f_1e[i], mu_1, s_1)
        cent_1_dx = (int_by_parts(f_1e[i+1], mu_1, s_1)
                     - int_by_parts(f_1e[i], mu_1, s_1))
        # (method dependent) integral dy:
        if method == "aug":
            # Equation (14)
            int_dy = norm.cdf(f_2e[i], mu_2, s_2)
            cent_2_dy = int_by_parts(f_2e[i], mu_2, s_2)
        elif method == "dom":
            # Equation (15)
            int_dy = norm.cdf(f_2e[i+1], mu_2, s_2)
            cent_2_dy = int_by_parts(f_2e[i+1], mu_2, s_2)
        elif method == "mix":
            # Average of equation (14) and (15) as discussed in the text below
            # equation (15)
            int_dy = (norm.cdf(f_2e[i], mu_2, s_2)
                      + norm.cdf(f_2e[i+1], mu_2, s_2)) / 2
            cent_2_dy = (int_by_parts(f_2e[i], mu_2, s_2)
                         + int_by_parts(f_2e[i+1], mu_2, s_2)) / 2
        else:
            raise NotImplementedError(f"Unknown method {method}")
        # Multiply the integrals and add to the variables
        PI += int_dx * int_dy
        centroid[:, 0] += cent_1_dx * int_dy
        centroid[:, 1] += int_dx * cent_2_dy
    # Final part is the integral dx from the x-coordinate of the right most Pareto
    # point to inf and the integral dy from -inf to the y-coordinate of the right
    # most Pareto point
    PI += (1 - norm.cdf(f_1e[-1], mu_1, s_1)) * norm.cdf(f_2e[-1], mu_2, s_2)
    # Integrate x dx by parts, where we can not use the helper function because of
    # different integration bounds (x-coordinate of right most Pareto point to inf)
    # and multiply by the integral dy from -inf to the y-coordinate of the right
    # most Pareto point. NOTE: mistake in the paper for the integral over dx where
    # mu * CDF is used instead of the correct mu * (1 - CDF)
    centroid[:, 0] += (mu_1 * (1 - norm.cdf(f_1e[-1], mu_1, s_1))
                       + s_1 * norm.pdf(f_1e[-1], mu_1, s_1)
                       ) * norm.cdf(f_2e[-1], mu_2, s_2)
    # Integrate dx same as in the PI and multiply by the integral y dy, which
    # again is performed using the helper function
    centroid[:, 1] += (
        1 - norm.cdf(f_1e[-1], mu_1, s_1)) * int_by_parts(f_2e[-1], mu_2, s_2)
    centroid /= np.maximum(PI[:, np.newaxis], 1e-10)
    return PI, centroid


def get_2D_EHVI(pred_mean: np.ndarray, pred_std: np.ndarray,
                pareto_points: np.ndarray, r: np.ndarray):
    """Calculates the two dimensional expected hypervolume improvement.

    Parameters
    ----------
    pred_mean : array_like, shape (N,2)
        the predicted mean from a ML model for both target properties
    pred_std : array_like, shape (N,2)
        the predicted std from a ML model for both target properties
    pareto_points : array_like, shape (M, 2)
        the points on the current Pareto front, must be correctly ordered
    r : array_like, shape (2,)
        reference point for hypervolume calculation

    Returns
    -------
    np.ndarray, shape (N,)
        array of the expected hypervolume improvement values
    """
    # Test that the pareto_points make a valid front (Important for the order of the
    # integrals)
    # The x values must be strictly ascending
    if any(np.diff(pareto_points[:, 0]) < 0.0):
        raise ValueError("The x values of the Pareto front must be in ascending order")
    # The corresponding y values must be strictly descending
    if any(np.diff(pareto_points[:, 1]) > 0.0):
        raise ValueError("The y values of the Pareto front must be in descending order")

    evhi = np.zeros(len(pred_mean))

    return evhi
