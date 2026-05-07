

import argparse
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def polyfit(x, y, degree):
    
    n = len(x)
    # build Vandermonde matrix
    A = np.zeros((n, degree + 1))
    for j in range(degree + 1):
        A[:, j] = x ** j

    # normal equations
    ATA = A.T @ A
    ATy = A.T @ y

    # solve via Gaussian elimination
    coeffs = gauss_solve(ATA, ATy)
    return coeffs


def gauss_solve(A, b):
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for col in range(n):
        # partial pivot
        max_row = col + np.argmax(np.abs(M[col:, col]))
        M[[col, max_row]] = M[[max_row, col]]

        for row in range(col + 1, n):
            factor = M[row, col] / M[col, col]
            M[row, col:] -= factor * M[col, col:]

    # back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - M[i, i+1:n] @ x[i+1:n]) / M[i, i]
    return x


def poly_eval(coeffs, x):
    result = np.zeros_like(x, dtype=float)
    for j, c in enumerate(coeffs):
        result += c * x ** j
    return result


def r_squared(y_true, y_pred):
    """Compute R² value."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def poly_to_str(coeffs):
    terms = []
    for j, c in enumerate(coeffs):
        if abs(c) < 1e-12:
            continue
        if j == 0:
            terms.append(f"{c:.4f}")
        elif j == 1:
            terms.append(f"{c:.4f}x")
        else:
            terms.append(f"{c:.4f}x^{j}")
    return " + ".join(terms) if terms else "0"



def cubic_spline(x, y):
    n = len(x) - 1
    h = np.diff(x)
    a = y.copy()

    # build tridiagonal system for c coefficients
    A_diag = np.zeros(n + 1)
    A_upper = np.zeros(n)
    A_lower = np.zeros(n)
    rhs = np.zeros(n + 1)

    A_diag[0] = 1.0
    A_diag[n] = 1.0

    for i in range(1, n):
        A_lower[i - 1] = h[i - 1]
        A_diag[i] = 2.0 * (h[i - 1] + h[i])
        A_upper[i] = h[i]
        rhs[i] = 3.0 * ((a[i + 1] - a[i]) / h[i] - (a[i] - a[i - 1]) / h[i - 1])

    # solve tridiagonal system (Thomas algorithm)
    c = tridiag_solve(A_lower, A_diag, A_upper, rhs)

    # compute b and d
    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])

    return a[:n], b, c[:n], d


def tridiag_solve(lower, diag, upper, rhs):
    n = len(diag)
    c = np.zeros(n)
    d = np.zeros(n)
    x = np.zeros(n)

    c[0] = upper[0] / diag[0] if len(upper) > 0 else 0
    d[0] = rhs[0] / diag[0]

    for i in range(1, n):
        lo = lower[i - 1] if i - 1 < len(lower) else 0
        up = upper[i] if i < len(upper) else 0
        m = diag[i] - lo * c[i - 1]
        c[i] = up / m if m != 0 else 0
        d[i] = (rhs[i] - lo * d[i - 1]) / m

    x[n - 1] = d[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def spline_eval(x_knots, a, b, c, d, x_eval):
    y_eval = np.zeros_like(x_eval, dtype=float)
    n = len(a)

    for k, xv in enumerate(x_eval):
        # find segment
        i = np.searchsorted(x_knots[:-1], xv, side='right') - 1
        i = max(0, min(i, n - 1))
        dx = xv - x_knots[i]
        y_eval[k] = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3

    return y_eval


def load_csv(filepath):
    dates = []
    prices = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        # find date and price columns
        header_lower = [h.strip().lower() for h in header]

        date_col = None
        price_col = None
        for i, h in enumerate(header_lower):
            if h in ('date', 'time', 'timestamp'):
                date_col = i
            if h in ('close', 'price', 'adj close', 'closing_price', 'value'):
                price_col = i

        if date_col is None:
            date_col = 0
        if price_col is None:
            price_col = 1

        date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y', '%Y/%m/%d']

        for row in reader:
            if len(row) <= max(date_col, price_col):
                continue
            date_str = row[date_col].strip()
            price_str = row[price_col].strip().replace(',', '')

            # try date formats
            parsed = None
            for fmt in date_formats:
                try:
                    parsed = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            if parsed is None:
                continue

            try:
                price = float(price_str)
            except ValueError:
                continue

            dates.append(parsed)
            prices.append(price)

    if len(dates) == 0:
        print("Error: no valid data found in CSV.", file=sys.stderr)
        sys.exit(1)

    # sort by date
    order = np.argsort(dates)
    dates = [dates[i] for i in order]
    prices = [prices[i] for i in order]

    return dates, np.array(prices, dtype=float)


def plot_results(dates, prices, x, coeffs, degree, r2, spline_data, forecast_days, output_path):
    """Generate the output plot."""
    x_knots, a, b, c, d = spline_data
    x_smooth = np.linspace(x[0], x[-1], 500)
    y_spline = spline_eval(x_knots, a, b, c, d, x_smooth)
    y_reg = poly_eval(coeffs, x_smooth)

    fig, ax = plt.subplots(figsize=(10, 5))

    # raw data
    ax.scatter(dates, prices, s=12, color='gray', alpha=0.5, label='Raw data', zorder=3)

    # spline
    smooth_dates = [dates[0] + timedelta(days=float(v)) for v in x_smooth]
    ax.plot(smooth_dates, y_spline, color='#2196F3', linewidth=1.5, label='Cubic spline')

    # regression
    ax.plot(smooth_dates, y_reg, color='#FF5722', linewidth=1.5, linestyle='--',
            label=f'Degree {degree} fit (R\u00B2={r2:.4f})')

    # forecast
    if forecast_days > 0:
        x_fore = np.linspace(x[-1], x[-1] + forecast_days, 50)
        y_fore = poly_eval(coeffs, x_fore)
        fore_dates = [dates[0] + timedelta(days=float(v)) for v in x_fore]
        ax.plot(fore_dates, y_fore, color='#FF5722', linewidth=1.5, linestyle=':',
                label=f'{forecast_days}-day forecast')
        ax.axvline(dates[-1], color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Stock/Crypto Price Predictor')
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_compare(dates, prices, x, results, output_path):
    """Plot multiple polynomial degrees for --compare mode."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(dates, prices, s=12, color='gray', alpha=0.5, label='Raw data', zorder=3)

    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    x_smooth = np.linspace(x[0], x[-1], 500)
    smooth_dates = [dates[0] + timedelta(days=float(v)) for v in x_smooth]

    for i, (deg, coeffs, r2) in enumerate(results):
        y_fit = poly_eval(coeffs, x_smooth)
        ax.plot(smooth_dates, y_fit, color=colors[i % len(colors)], linewidth=1.5,
                label=f'Degree {deg} (R\u00B2={r2:.4f})')

    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Polynomial Degree Comparison')
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Stock/Crypto Price Predictor')
    parser.add_argument('csv_file', help='CSV file with date and price columns')
    parser.add_argument('--degree', type=int, default=2, help='Polynomial degree (1-4, default 2)')
    parser.add_argument('--forecast', type=int, default=0, help='Number of days to forecast')
    parser.add_argument('--compare', action='store_true', help='Compare polynomial degrees 1-4')
    parser.add_argument('--output', default='output.png', help='Output plot filename')
    args = parser.parse_args()

    if args.degree < 1 or args.degree > 4:
        print("Error: degree must be 1-4.", file=sys.stderr)
        sys.exit(1)

    # load data
    dates, prices = load_csv(args.csv_file)
    print(f"Loaded {len(prices)} data points")

    # convert dates to numeric x (days from start)
    x = np.array([(d - dates[0]).days for d in dates], dtype=float)

    if args.compare:
        # compare mode
        results = []
        best_deg, best_r2 = 1, -1
        for deg in range(1, 5):
            coeffs = polyfit(x, prices, deg)
            y_pred = poly_eval(coeffs, x)
            r2 = r_squared(prices, y_pred)
            results.append((deg, coeffs, r2))
            print(f"  Degree {deg}: {poly_to_str(coeffs)}")
            print(f"           R\u00B2 = {r2:.6f}")
            if r2 > best_r2:
                best_deg, best_r2 = deg, r2

        print(f"\nBest fit: degree {best_deg} (R\u00B2 = {best_r2:.6f})")
        plot_compare(dates, prices, x, results, args.output)
        print(f"Plot saved to {args.output}")

    else:
        coeffs = polyfit(x, prices, args.degree)
        y_pred = poly_eval(coeffs, x)
        r2 = r_squared(prices, y_pred)

        print(f"Fit equation: {poly_to_str(coeffs)}")
        print(f"R\u00B2 = {r2:.6f}")

        a, b, c, d = cubic_spline(x, prices)
        spline_data = (x, a, b, c, d)

        if args.forecast > 0:
            x_fore = np.linspace(x[-1] + 1, x[-1] + args.forecast, args.forecast)
            y_fore = poly_eval(coeffs, x_fore)
            print(f"\n{args.forecast}-day forecast:")
            for i in range(min(args.forecast, 10)):
                fore_date = dates[-1] + timedelta(days=int(x_fore[i] - x[-1]))
                print(f"  {fore_date.strftime('%Y-%m-%d')}: ${y_fore[i]:.2f}")
            if args.forecast > 10:
                print(f"  ... ({args.forecast - 10} more days)")

        plot_results(dates, prices, x, coeffs, args.degree, r2, spline_data,
                     args.forecast, args.output)
        print(f"Plot saved to {args.output}")


if __name__ == '__main__':
    main()
