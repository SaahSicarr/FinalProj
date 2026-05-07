## What it does

Takes a CSV of historical price data, fits it using least-squares polynomial regression and cubic spline interpolation, and produces a visualization with optional forecast.

## Numerical Methods

- **Least-squares polynomial regression** — normal equations solved via Gaussian elimination with partial pivoting
- **Natural cubic spline interpolation** — tridiagonal system solved via Thomas algorithm


# basic fit (degree 2 by default)
python price_predictor.py sample_data.csv

# change polynomial degree
python price_predictor.py data.csv --degree 3

# forecast 7 days ahead
python price_predictor.py data.csv --forecast 7

# compare degrees 1-4
python price_predictor.py data.csv --compare

# custom output filename
python price_predictor.py data.csv --output my_plot.png
```

## CSV Format

The CSV needs a header row with a date column and a price column. These names are auto-detected:
- Date columns: `Date`, `Time`, `Timestamp`
- Price columns: `Close`, `Price`, `Adj Close`, `Value`

## Testing

pip install pytest
py -m pytest test_price_predictor.py -v
