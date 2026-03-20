import pytest
import pandas as pd
import numpy as np
from scripts.build_value_features_v2 import safe_divide


class TestSafeDivide:
    def test_scalar_denom_zero(self):
        numer = pd.Series([1, 2, 3])
        result = safe_divide(numer, 0)
        expected = pd.Series([0.0, 0.0, 0.0])
        pd.testing.assert_series_equal(result, expected)

    def test_scalar_denom_nonzero(self):
        numer = pd.Series([10, 20, 30])
        result = safe_divide(numer, 2)
        expected = pd.Series([5.0, 10.0, 15.0])
        pd.testing.assert_series_equal(result, expected)

    def test_series_denom_with_zeros(self):
        numer = pd.Series([10, 20, 30])
        denom = pd.Series([2, 0, 5])
        result = safe_divide(numer, denom)
        expected = pd.Series([5.0, 0.0, 6.0])
        pd.testing.assert_series_equal(result, expected)

    def test_series_denom_no_zeros(self):
        numer = pd.Series([10, 20, 30])
        denom = pd.Series([2, 4, 5])
        result = safe_divide(numer, denom)
        expected = pd.Series([5.0, 5.0, 6.0])
        pd.testing.assert_series_equal(result, expected)

    def test_inf_and_nan_handling(self):
        numer = pd.Series([1, 2, 3])
        denom = pd.Series([0, np.inf, np.nan])
        result = safe_divide(numer, denom)
        expected = pd.Series([0.0, 0.0, 0.0])
        pd.testing.assert_series_equal(result, expected)