"""
Tests for Stock/Crypto Price Predictor
"""

import os
import csv
import time
import subprocess
import numpy as np
import pytest
from price_predictor import (
    polyfit, poly_eval, r_squared, gauss_solve,
    cubic_spline, spline_eval, load_csv
)


def make_csv(filepath, dates, prices, date_col='Date', price_col='Close'):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([date_col, price_col])
        for d, p in zip(dates, prices):
            writer.writerow([d, f"{p:.4f}"])


@pytest.fixture
def tmp_csv(tmp_path):
    """Generate a simple linear CSV for basic tests."""
    filepath = tmp_path / "test.csv"
    dates = [f"2025-01-{i+1:02d}" for i in range(30)]
    prices = [100 + 2 * i + np.random.normal(0, 0.3) for i in range(30)]
    make_csv(filepath, dates, prices)
    return str(filepath)


@pytest.fixture
def linear_csv(tmp_path):
    """Linear dataset: y = 2x + 5 with small noise."""
    filepath = tmp_path / "linear.csv"
    np.random.seed(42)
    dates = [f"2025-01-{i+1:02d}" for i in range(30)]
    x = np.arange(30, dtype=float)
    prices = 2.0 * x + 5.0 + np.random.normal(0, 0.2, 30)
    make_csv(filepath, dates, prices)
    return str(filepath), 2.0, 5.0 


@pytest.fixture
def quadratic_csv(tmp_path):
    """Perfectly quadratic dataset."""
    filepath = tmp_path / "quad.csv"
    dates = [f"2025-01-{i+1:02d}" for i in range(30)]
    x = np.arange(30, dtype=float)
    prices = 0.5 * x**2 + 3.0 * x + 10.0
    make_csv(filepath, dates, prices)
    return str(filepath)


@pytest.fixture
def large_csv(tmp_path):
    """365-row CSV for performance test."""
    filepath = tmp_path / "large.csv"
    from datetime import datetime, timedelta
    base = datetime(2025, 1, 1)
    dates = [(base + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(365)]
    np.random.seed(0)
    prices = 100 + np.cumsum(np.random.normal(0, 1, 365))
    make_csv(filepath, dates, prices)
    return str(filepath)


#  Regression accuracy on linear data

def test_r1_regression_linear(linear_csv):
    filepath, true_slope, true_intercept = linear_csv
    dates, prices = load_csv(filepath)
    x = np.arange(len(prices), dtype=float)
    coeffs = polyfit(x, prices, 1)

    # coeffs[0] = intercept, coeffs[1] = slope
    assert abs(coeffs[1] - true_slope) < 0.1, f"Slope {coeffs[1]:.4f} not within 0.1 of {true_slope}"
    assert abs(coeffs[0] - true_intercept) < 0.5, f"Intercept {coeffs[0]:.4f} not within 0.5 of {true_intercept}"


# Spline passes through all data points 

def test_r2_spline_passthrough(tmp_csv):
    dates, prices = load_csv(tmp_csv)
    x = np.arange(len(prices), dtype=float)
    a, b, c, d = cubic_spline(x, prices)
    y_at_knots = spline_eval(x, a, b, c, d, x)

    for i in range(len(prices)):
        assert abs(y_at_knots[i] - prices[i]) < 1e-10, \
            f"Spline mismatch at point {i}: {y_at_knots[i]:.12f} vs {prices[i]:.12f}"


# R^2 matches NumPy 

def test_r3_r2_matches_numpy(tmp_csv):
    dates, prices = load_csv(tmp_csv)
    x = np.arange(len(prices), dtype=float)

    for deg in [1, 2, 3]:
        coeffs = polyfit(x, prices, deg)
        y_pred = poly_eval(coeffs, x)
        our_r2 = r_squared(prices, y_pred)

        np_coeffs = np.polyfit(x, prices, deg)
        np_pred = np.polyval(np_coeffs, x)
        np_r2 = 1 - np.sum((prices - np_pred)**2) / np.sum((prices - np.mean(prices))**2)

        assert abs(our_r2 - np_r2) < 0.001, \
            f"Degree {deg}: our R²={our_r2:.6f} vs NumPy R²={np_r2:.6f}"


# Quadratic dataset gets R² > 0.999

def test_r4_quadratic_fit(quadratic_csv):
    filepath = quadratic_csv
    dates, prices = load_csv(filepath)
    x = np.arange(len(prices), dtype=float)
    coeffs = polyfit(x, prices, 2)
    y_pred = poly_eval(coeffs, x)
    r2 = r_squared(prices, y_pred)

    assert r2 > 0.999, f"R² = {r2:.6f}, expected > 0.999"


# Processing speed

def test_r5_performance(large_csv):
    filepath = large_csv
    dates, prices = load_csv(filepath)
    x = np.arange(len(prices), dtype=float)

    start = time.time()
    coeffs = polyfit(x, prices, 2)
    poly_eval(coeffs, x)
    r_squared(prices, poly_eval(coeffs, x))
    a, b, c, d = cubic_spline(x, prices)
    spline_eval(x, a, b, c, d, np.linspace(x[0], x[-1], 500))
    elapsed = time.time() - start

    assert elapsed < 2.0, f"Processing took {elapsed:.2f}s, expected < 2s"


# No crash on valid input

def test_r6_no_crash(tmp_csv, tmp_path):
    output = str(tmp_path / "test_output.png")
    result = subprocess.run(
        ['python', 'price_predictor.py', tmp_csv, '--output', output],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    assert result.returncode == 0, f"Program crashed: {result.stderr}"
    assert os.path.exists(output), "PNG not created"
    assert 'R²' in result.stdout or 'R\u00b2' in result.stdout, "R² not printed"


# Compare mode picks best degree

def test_r7_compare_mode(quadratic_csv, tmp_path):
    filepath = quadratic_csv
    output = str(tmp_path / "compare_output.png")
    result = subprocess.run(
        ['python', 'price_predictor.py', filepath, '--compare', '--output', output],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    assert result.returncode == 0, f"Program crashed: {result.stderr}"
    # degree 2 should be best for quadratic data
    assert 'Best fit: degree 2' in result.stdout, f"Expected degree 2 as best, got: {result.stdout}"
