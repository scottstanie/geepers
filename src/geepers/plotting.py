import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def create_rate_comparison_plot(
    rates: pd.DataFrame,
    quality_column: str = "similarity",
    quality_cmap: str = "viridis",
) -> tuple[plt.Figure, plt.Axes]:
    """Create a publication-quality scatter plot comparing GPS and InSAR rates.

    Parameters
    ----------
    rates : pd.DataFrame
        DataFrame with columns: station, gps_velocity, insar_velocity

    Returns
    -------
    fig : plt.Figure
        The figure object
    ax : plt.Axes
        The axes object
    """
    plt.style.use("seaborn-v0_8-paper")

    # Calculate correlation coefficient and RMSE
    corr_coef = stats.pearsonr(rates.insar_velocity, rates.gps_velocity)[0]
    rmse = np.sqrt(np.mean((rates.gps_velocity - rates.insar_velocity) ** 2))

    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    # Plot the scatter points
    scatter_img = ax.scatter(
        rates.insar_velocity,
        rates.gps_velocity,
        # color="#2b8cbe",
        c=rates[quality_column],
        cmap=quality_cmap,
        alpha=0.8,
        s=40,
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

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    label = f"Corr coeff: {corr_coef:.2f}\nRMSE: {rmse:.1f} mm/yr"
    ax.text(
        x=0.05,
        y=0.95,
        s=label,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
    )
    fig.colorbar(scatter_img, ax=ax, label=quality_column)
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
#     (abs(rates.gps_velocity) > extreme_threshold)
#     | (abs(rates.insar_velocity) > extreme_threshold)
# ]
# if not extreme_rates.empty:
#     print("\nStations with extreme rates (>15 mm/yr):")
#     print(extreme_rates)
