from typing import Any

try:
    import jax.numpy as jnp
    from dolphin.timeseries import (
        least_absolute_deviations,
    )  # TODO: port just this part
    from jax.typing import ArrayLike
except ImportError:
    print("dolphin not installed, cannot use robust_linear_fit")
    # import warnings
    # warnings.warn("dolphin not installed, cannot use robust_linear_fit", stacklevel=2)
    ArrayLike = Any
    import numpy as jnp


def robust_linear_fit(
    x: ArrayLike,
    y: ArrayLike,
    w: ArrayLike | None = None,
    return_cov: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Perform robust linear fitting using L1 norm (Least Absolute Deviations) via ADMM.

    This function combines traditional linear fitting with ADMM-based L1 optimization
    to provide robust estimates less sensitive to outliers.

    Parameters
    ----------
    x : ArrayLike
        Independent variable data points. Shape (M,)
    y : ArrayLike
        Dependent variable data points. Shape (M,) or (M, K)
    w : ArrayLike | None, optional
        Weights for weighted fitting. Shape (M,)
    return_cov : bool, optional
        Whether to return the covariance matrix, by default True

    Returns
    -------
    coefficients : jnp.ndarray
        The solution vector (N,) where N is the polynomial order + 1
    covariance : jnp.ndarray, optional
        The covariance matrix (N, N), returned if return_cov=True

    """
    deg = 1
    order = deg + 1

    # Validate inputs
    x_arr, y_arr = jnp.asarray(x), jnp.asarray(y)
    if x_arr.ndim != 1:
        msg = "expected 1D vector for x"
        raise TypeError(msg)
    if y_arr.ndim not in (1, 2):
        msg = "expected 1D or 2D array for y"
        raise TypeError(msg)
    if x_arr.shape[0] != y_arr.shape[0]:
        msg = "expected x and y to have same length"
        raise TypeError(msg)

    # Set up design matrix A
    A = jnp.vander(x_arr, order)

    # Apply weights if provided
    if w is not None:
        w_arr = jnp.asarray(w)
        if w_arr.ndim != 1:
            msg = "expected a 1-d array for weights"
            raise TypeError(msg)
        if w_arr.shape[0] != y_arr.shape[0]:
            msg = "expected w and y to have the same length"
            raise TypeError(msg)
        A *= w_arr[:, jnp.newaxis]
        if y_arr.ndim == 2:
            y_arr = y_arr * w_arr[:, jnp.newaxis]
        else:
            y_arr = y_arr * w_arr

    # Compute Cholesky factor for ADMM
    R = jnp.linalg.cholesky(A.T @ A)
    x, resid = least_absolute_deviations(A, y_arr, R, max_iter=50)

    if not return_cov:
        return x

    # Compute covariance matrix
    Vbase = jnp.linalg.inv(A.T @ A)
    return x, Vbase
