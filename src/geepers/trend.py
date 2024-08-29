from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import datetime_to_float


@dataclass
class TrendEstimator:
    series: pd.Series
    tol_days: int = 30

    def tsia(self):
        """Calculate the Thiel-Sen Inter-annual slope of a data Series

        https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator

        Assumes the `series` has a DatetimeIndex.
        Forms all possible difference of data which span 1 year +/- `tol_days`,
        then takes the median slope of these differences
        """
        # Get the non-nan values of the series
        data = self.series.dropna().values
        times = self.series.dropna().index
        # Convert to numerical values for fitting:
        # t = (times - times[0]).days
        t = datetime_to_float(times)
        time_diffs = self._get_all_differences(t)
        slopes = self._get_all_differences(data) / time_diffs

        # Now pick slopes within `tol_days` of annual
        # > 180 to make sure we dont' use super short periods
        accept_idxs = np.logical_and(
            time_diffs > 180, (self._dist_from_year(time_diffs) < self.tol_days)
        )
        slopes_annual = slopes[accept_idxs]
        slope = np.median(slopes_annual)

        # Add Normal dist. factor to MAD
        sig = 1.4826 * self.mad(slopes_annual)
        # TODO: track down Ben for origina of this formula... prob on Wiki for TSIA
        uncertainty = 3 * np.sqrt(np.pi / 2) * sig / np.sqrt(len(slopes_annual) / 4)
        b = np.median(data - slope * t)
        return slope, b, uncertainty

    @staticmethod
    def mad(x):
        """Median absolut deviation"""
        return np.median(np.abs(x - np.median(x)))

    @staticmethod
    def _dist_from_year(v):
        """Get the number of days away from 365, mod 1 year"""
        return np.abs((v + 180) % 365 - 180)

    @staticmethod
    def _get_all_differences(a):
        """Calculate all possible differences between elements of `a`"""
        n = len(a)
        x = np.reshape(a, (1, n))
        difference_matrix = x - x.transpose()
        # Now get the upper half (bottom is redundant)
        return difference_matrix[np.triu_indices(n)].ravel()


def _fit_line_to_dates(df):
    return np.array([linear_trend(df[col]).tail(1).squeeze() for col in df.columns])


def fit_line(series, median=False):
    """Fit a line to `series` with (possibly) uneven dates as index.

    Can be used to detrend, or predict final value

    Args:
        series (pd.Series): data to fit, with a DatetimeIndex
        median (bool): if true, use the TSIA median estimator to fit

    Returns: [slope, intercept]
    """
    # TODO: check that subtracting first item doesn't change it

    series_clean = series.dropna()
    idxs = datetime_to_float(series_clean.index)

    coeffs = np.polyfit(idxs, series_clean, 1)
    if median:
        # Replace the Least squares fit with the median inter-annual slope
        est = TrendEstimator(series)
        # med_slope, intercept, uncertainty = est.tsia()
        coeffs = est.tsia()[:2]
    return coeffs


def linear_trend(series=None, coeffs=None, index=None, x=None, median=False):
    """Get a series of points representing a linear trend through `series`

    First computes the lienar regression, the evaluates at each
    dates of `series.index`

    Args:
        series (pandas.Series): data with DatetimeIndex as the index.
        coeffs (array or List): [slope, intercept], result from np.polyfit
        index (DatetimeIndex, list[date]): Optional. If not passing series, can pass
            the DatetimeIndex or list of dates to evaluate coeffs at.
            Converts to numbers using `matplotlib.dates.date2num`
        x (ndarray-like): directly pass the points to evaluate the poly1d
    Returns:
        Series: a line, equal length to arr, with same index as `series`
    """
    if coeffs is None:
        coeffs = fit_line(series, median=median)

    if index is None and x is None:
        index = series.dropna().index
    if x is None:
        x = datetime_to_float(index)

    poly = np.poly1d(coeffs)
    linear_points = poly(x)
    return pd.Series(linear_points, index=index)


def _flat_std(series):
    """Find the std dev of an Series with a linear component removed"""
    return np.std(series - linear_trend(series))


def moving_average(arr, window_size=7):
    """Takes a 1D array and returns the running average of same size"""
    if not window_size:
        return arr
    # return uniform_filter1d(arr, size=window_size, mode='nearest')
    return np.array(pd.Series(arr).rolling(window_size).mean())
