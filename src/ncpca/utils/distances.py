"""Calculate distances between point clouds."""

import numpy as np
from typing import Optional, Tuple
from scipy.linalg import sqrtm


def fit_gaussian(
    x: np.ndarray, max_dim: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a multivariate Gaussian to the given dataset, and return the mean and
    covariance matrix.

    :param max_dim: restrict to the top `max_dim` dimensions (unless it's `None`)
    """
    if max_dim is not None:
        x = x[:, :max_dim]

    mu = np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    return mu, cov


def dkl_gaussian(
    gauss1: Tuple[np.ndarray, np.ndarray], gauss2: Tuple[np.ndarray, np.ndarray]
) -> float:
    """Find the KL divergence between two Gaussians, encoded as tuples of the form
    `(mean, covariance_matrix)`.
    """
    det1 = np.linalg.det(gauss1[1])
    det2 = np.linalg.det(gauss2[1])

    dmu = gauss1[0] - gauss2[0]

    inv_cov2 = np.linalg.inv(gauss2[1])

    k = len(gauss1[0])
    log_term = np.log(det2 / det1) - k

    quadratic_term = np.dot(dmu, inv_cov2 @ dmu)

    trace_term = np.trace(inv_cov2 @ gauss1[1])

    return 0.5 * (log_term + quadratic_term + trace_term)


def dkl_symm_gaussian(
    gauss1: Tuple[np.ndarray, np.ndarray], gauss2: Tuple[np.ndarray, np.ndarray]
) -> float:
    """Find the symmetrized KL divergence between two Gaussians, encoded as tuples
    `(mean, covariance_matrix)`.
    """
    det1 = np.linalg.det(gauss1[1])
    det2 = np.linalg.det(gauss2[1])

    dmu = gauss1[0] - gauss2[0]

    inv_cov1 = np.linalg.inv(gauss1[1])
    inv_cov2 = np.linalg.inv(gauss2[1])

    quadratic_term = np.dot(dmu, 0.5 * (inv_cov1 + inv_cov2) @ dmu)
    trace_term = 0.5 * (np.trace(inv_cov2 @ gauss1[1]) + np.trace(inv_cov1 @ gauss2[1]))

    k = len(gauss1[0])
    return 0.5 * (quadratic_term + trace_term - k)


def db_gaussian(
    gauss1: Tuple[np.ndarray, np.ndarray], gauss2: Tuple[np.ndarray, np.ndarray]
) -> float:
    """Find the Bhattacharyya distance between two Gaussians, enocded as tuples
    `(mean, covariance_matrix)`.
    """
    cov = 0.5 * (gauss1[1] + gauss2[1])
    inv_cov = np.linalg.inv(cov)

    dmu = gauss1[0] - gauss2[0]

    det1 = np.linalg.det(gauss1[1])
    det2 = np.linalg.det(gauss2[1])
    det = np.linalg.det(cov)

    quadratic_term = (1 / 4) * np.dot(dmu, inv_cov @ dmu)
    log_term = np.log(det / np.sqrt(det1 * det2))

    return 0.5 * (log_term + quadratic_term)


def bc_gaussian(
    gauss1: Tuple[np.ndarray, np.ndarray], gauss2: Tuple[np.ndarray, np.ndarray]
) -> float:
    """Find the Bhattacharyya coefficient between two Gaussians, enocded as tuples
    `(mean, covariance_matrix)`.
    """
    db = db_gaussian(gauss1, gauss2)
    return np.exp(-db)


def dhellinger_gaussian(
    gauss1: Tuple[np.ndarray, np.ndarray], gauss2: Tuple[np.ndarray, np.ndarray]
) -> float:
    """Find the Hellinger distance between two Gaussians, enocded as tuples of the form
    `(mean, covariance_matrix)`.
    """
    bc = bc_gaussian(gauss1, gauss2)
    return np.sqrt(1 - bc)


def dwasserstein_gaussian(
    gauss1: Tuple[np.ndarray, np.ndarray], gauss2: Tuple[np.ndarray, np.ndarray]
) -> float:
    """Find the Wasserstein distance between two Gaussians, enocded as tuples
    `(mean, covariance_matrix)`.
    """
    dmu = gauss1[0] - gauss2[0]

    sqrt_cov2 = sqrtm(gauss2[1])
    sqrt_cov_term = sqrtm(sqrt_cov2 @ gauss1[1] @ sqrt_cov2)

    quadratic_term = np.sum(dmu**2)
    trace_term = np.trace(gauss1[1] + gauss2[1] - 2 * sqrt_cov_term)

    return quadratic_term + trace_term


def cloud_distance(
    x: np.ndarray, y: np.ndarray, kind: str = "kl_symm", **kwargs
) -> float:
    """Calculate a (pseudo-)distance between the Gaussian approximations to two
    datasets.

    The `max_dim` argument is forwarded to `fit_gaussian`.

    :param kind: can be `"kl"`, `"kl_symm"`, `"b"` (for Bhattacharyya distance),
        `"hellinger"`, or `"wasserstein"`
    :param **kwargs: additional keyword arguments are passed to `fit_gaussian`
    """
    gauss1 = fit_gaussian(x, **kwargs)
    gauss2 = fit_gaussian(y, **kwargs)

    fct_mapping = {
        "kl": dkl_gaussian,
        "kl_symm": dkl_symm_gaussian,
        "bc": db_gaussian,
        "hellinger": dhellinger_gaussian,
        "wasserstein": dwasserstein_gaussian,
    }
    fct = fct_mapping[kind]

    return fct(gauss1, gauss2)
