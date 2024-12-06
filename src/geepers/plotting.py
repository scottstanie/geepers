import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats


def create_comparison_plot(df: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """Create a publication-quality scatter plot comparing GPS and InSAR measurements.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'station', 'date', 'measurement', and 'value' columns

    Returns
    -------
    fig : plt.Figure
        The figure object
    ax : plt.Axes
        The axes object
    """
    # Set the style for publication-quality plots
    plt.style.use("seaborn-v0_8-paper")

    # Pivot the data to get GPS and InSAR measurements
    df_wide = df.pivot_table(
        index=["station", "date"], columns="measurement", values="value"
    ).reset_index()

    # Calculate correlation coefficient and RMSE
    valid_mask = ~(np.isnan(df_wide.los_gps) | np.isnan(df_wide.los_insar))
    gps_valid = df_wide.los_gps[valid_mask]
    insar_valid = df_wide.los_insar[valid_mask]

    corr_coef = stats.pearsonr(gps_valid, insar_valid)[0]
    rmse = np.sqrt(np.mean((gps_valid - insar_valid) ** 2))

    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    # Plot the scatter points
    ax.scatter(
        df_wide.los_insar,
        df_wide.los_gps,
        color="#2b8cbe",
        alpha=0.6,
        s=30,
        label=f"Corr coeff: {corr_coef:.2f}\nRMSE: {rmse:.1f} mm/yr",
    )

    # Plot the 1:1 line
    lims = np.array(
        [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
    )
    ax.plot(lims, lims, "gray", linestyle="-", alpha=0.8, zorder=0)

    # Set equal aspect ratio
    ax.set_aspect("equal")

    # Set limits to be symmetric around zero
    max_val = max(abs(lims))
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    # Add grid
    ax.grid(True, linestyle=":", alpha=0.6)

    # Set labels and title
    ax.set_xlabel("InSAR [mm/year]")
    ax.set_ylabel("GPS [mm/year]")

    # Add minor ticks
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    # Add legend
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        bbox_to_anchor=(0.02, 0.98),
        loc="upper left",
    )

    # Adjust layout
    plt.tight_layout()

    return fig, ax


# # Read and process the data
# df = pd.read_csv("combined_data.csv")
# fig, ax = create_comparison_plot(df)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats
from datetime import datetime


def calculate_rates(
    df: pd.DataFrame, outlier_threshold: float = 50, to_mm: bool = True
) -> pd.DataFrame:
    """Calculate rates for each station from GPS and InSAR time series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: station, date, measurement, value
    outlier_threshold : float
        Remove measurements with absolute values greater than this

    Returns
    -------
    pd.DataFrame
        DataFrame with GPS and InSAR rates for each station
        If `to_mm` is True, output is in mm/year.
        Otherwise, units are no changed (meters/year)
    """
    # Convert date to datetime if it's not already
    df["date"] = pd.to_datetime(df["date"])

    # Remove obvious outliers
    df = df[abs(df["value"]) < outlier_threshold]

    # Pivot to get separate GPS and InSAR columns
    df_wide = df.pivot_table(
        index=["station", "date"], columns="measurement", values="value"
    ).reset_index()

    # Function to calculate rate for a single station's time series
    def calc_station_rate(group):
        # Convert dates to years since first measurement
        years = (group["date"] - group["date"].min()).dt.total_seconds() / (
            365.25 * 24 * 3600
        )
        time_span_years = years.iloc[-1] - years.iloc[0]

        # Calculate rates using least squares fit
        num_gps = gps_rate = insar_rate = insar_rate_l2 = tcoh = similarity = np.nan
        const = 1000 if to_mm else 1

        # GPS rate
        if not group["los_gps"].isna().all():
            mask = ~group["los_gps"].isna()
            if sum(mask) > 2:  # Need at least 3 points for meaningful rate
                gps_rate = np.polyfit(years[mask], group["los_gps"][mask], 1)[0] * const
                num_gps = len(group["los_gps"][mask])

        # InSAR rate
        if not group["los_insar"].isna().all():
            mask = ~group["los_insar"].isna()
            if sum(mask) > 2:  # Need at least 3 points for meaningful rate
                insar_rate_l2 = (
                    np.polyfit(years[mask], group["los_insar"][mask], 1)[0] * const
                )
                insar_rate = (
                    robust_linear_fit(
                        years[mask].values, group["los_insar"][mask].values
                    )[0]
                    * const
                )

        if not np.isnan(insar_rate):
            tcoh = group["temporal_coherence"].dropna().iloc[0]
            similarity = group["similarity"].dropna().iloc[0]
        return pd.Series(
            {
                "gps_rate": gps_rate,
                "insar_rate": np.asarray(insar_rate),
                "insar_rate_l2": np.asarray(insar_rate_l2),
                "difference": np.asarray(insar_rate - gps_rate),
                "temporal_coherence": tcoh,
                "similarity": similarity,
                "num_gps_points": num_gps,
                "gps_time_span_years": time_span_years,
            }
        )

    # Calculate rates for each station
    rates = df_wide.groupby("station").apply(calc_station_rate).reset_index()

    # Remove stations where either rate is NaN
    rates = rates.dropna()
    return rates


def create_rate_comparison_plot(
    rates: pd.DataFrame,
    quality_column: str = "similarity",
    quality_cmap: str = "viridis",
) -> tuple[plt.Figure, plt.Axes]:
    """Create a publication-quality scatter plot comparing GPS and InSAR rates.

    Parameters
    ----------
    rates : pd.DataFrame
        DataFrame with columns: station, gps_rate, insar_rate

    Returns
    -------
    fig : plt.Figure
        The figure object
    ax : plt.Axes
        The axes object
    """
    plt.style.use("seaborn-v0_8-paper")

    # Calculate correlation coefficient and RMSE
    corr_coef = stats.pearsonr(rates.insar_rate, rates.gps_rate)[0]
    rmse = np.sqrt(np.mean((rates.gps_rate - rates.insar_rate) ** 2))

    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    # Plot the scatter points
    scatter_img = ax.scatter(
        rates.insar_rate,
        rates.gps_rate,
        # color="#2b8cbe",
        c=rates[quality_column],
        cmap=quality_cmap,
        alpha=0.8,
        s=40,
        label=f"Corr coeff: {corr_coef:.2f}\nRMSE: {rmse:.1f} mm/yr",
    )

    # Plot the 1:1 line
    lims = np.array(
        [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
    )
    ax.plot(lims, lims, "gray", linestyle="-", alpha=0.8, zorder=0)

    # Set equal aspect ratio
    ax.set_aspect("equal")

    # Set limits to be symmetric around zero
    max_val = max(abs(lims))
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    # Add grid
    ax.grid(True, linestyle=":", alpha=0.6)

    # Set labels and title
    ax.set_xlabel("InSAR Rate [mm/year]")
    ax.set_ylabel("GPS Rate [mm/year]")

    # Add minor ticks
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    # # Add legend
    # ax.legend(
    #     frameon=True,
    #     facecolor="white",
    #     edgecolor="none",
    #     bbox_to_anchor=(0.02, 0.98),
    #     loc="upper left",
    # )
    fig.colorbar(scatter_img, ax=ax)

    # Adjust layout
    plt.tight_layout()

    return fig, ax


# # Read and process the data
# df = pd.read_csv("combined_data.csv")
# rates = calculate_rates(df)
# print(f"Found rates for {len(rates)} stations")

# # Create and save the plot
# fig, ax = create_rate_comparison_plot(rates)
# plt.savefig("gps_insar_rates_comparison.pdf", bbox_inches="tight", dpi=300)
# plt.savefig("gps_insar_rates_comparison.png", bbox_inches="tight", dpi=300)

# # Print stations with extreme rates for inspection
# extreme_threshold = 15  # mm/yr
# extreme_rates = rates[
#     (abs(rates.gps_rate) > extreme_threshold)
#     | (abs(rates.insar_rate) > extreme_threshold)
# ]
# if not extreme_rates.empty:
#     print("\nStations with extreme rates (>15 mm/yr):")
#     print(extreme_rates)

from functools import partial
import jax
import jax.numpy as jnp
from typing import Union, Tuple, TypeVar
from jaxtyping import ArrayLike

ArrayLike = TypeVar("ArrayLike", jnp.ndarray, float)


# @partial(jax.jit, static_argnames=["return_cov", "max_iter"])
def robust_linear_fit(
    x: ArrayLike,
    y: ArrayLike,
    w: ArrayLike | None = None,
    return_cov: bool = False,
    rho: float = 0.4,
    alpha: float = 1.0,
    max_iter: int = 20,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
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
    rho : float, optional
        Augmented Lagrangian parameter, by default 0.4
    alpha : float, optional
        Over-relaxation parameter (typically between 1.0 and 1.8), by default 1.0
    max_iter : int, optional
        Maximum number of ADMM iterations, by default 20

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
        raise TypeError("expected 1D vector for x")
    if y_arr.ndim not in (1, 2):
        raise TypeError("expected 1D or 2D array for y")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise TypeError("expected x and y to have same length")

    # Set up design matrix A
    A = jnp.vander(x_arr, order)

    # Apply weights if provided
    if w is not None:
        w_arr = jnp.asarray(w)
        if w_arr.ndim != 1:
            raise TypeError("expected a 1-d array for weights")
        if w_arr.shape[0] != y_arr.shape[0]:
            raise TypeError("expected w and y to have the same length")
        A *= w_arr[:, jnp.newaxis]
        if y_arr.ndim == 2:
            y_arr = y_arr * w_arr[:, jnp.newaxis]
        else:
            y_arr = y_arr * w_arr

    # Compute Cholesky factor for ADMM
    R = jnp.linalg.cholesky(A.T @ A)
    from dolphin.timeseries import least_absolute_deviations

    x, resid = least_absolute_deviations(A, y_arr, R, max_iter=50)

    if not return_cov:
        return x

    # Compute covariance matrix
    Vbase = jnp.linalg.inv(A.T @ A)
    return x, Vbase
