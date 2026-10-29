"""Market data: download from Yahoo Finance, cache to CSV, or generate synthetic data for offline testing."""
import os

import numpy as np
import pandas as pd

import config as C


def synthetic_prices(n=3000, seed=C.SEED):
    """Fake but realistic prices: fat tails, volatility regimes, correlations and slowly drifting trends.
    Only for testing the code offline. Results on this data say nothing about real markets."""
    rng = np.random.default_rng(seed)
    names = list(C.TICKERS)
    k = len(names)
    vol = np.array([0.16, 0.06, 0.32, 0.15, 0.08, 0.15])
    mean = np.array([0.08, 0.02, 0.03, 0.04, 0.00, 0.09])
    corr = np.array([[1.0, -0.3, 0.4, 0.0, 0.2, 0.6], [-0.3, 1.0, -0.2, 0.2, 0.0, -0.2],
                     [0.4, -0.2, 1.0, 0.1, 0.2, 0.3], [0.0, 0.2, 0.1, 1.0, 0.3, 0.0],
                     [0.2, 0.0, 0.2, 0.3, 1.0, 0.1], [0.6, -0.2, 0.3, 0.0, 0.1, 1.0]])
    L = np.linalg.cholesky(corr)
    stress = np.zeros(n, dtype=bool)                       # two-state volatility regime (calm / stress)
    for t in range(1, n):
        stress[t] = rng.random() < (0.95 if stress[t - 1] else 0.01)
    mult = np.where(stress, 2.2, 1.0)[:, None]
    z = rng.standard_t(5, (n, k)) / np.sqrt(5 / 3) @ L.T   # unit-variance fat-tailed shocks
    phi = 0.988                                            # slowly varying drift -> some trend to detect
    trend = np.zeros((n, k))
    for t in range(1, n):
        trend[t] = phi * trend[t - 1] + rng.standard_normal(k) * 6.1e-5
    r = mean / C.TRADING_DAYS + trend + vol / np.sqrt(C.TRADING_DAYS) * mult * z
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=names)


def load_prices(synthetic=False, refresh=False):
    os.makedirs(C.DATA_DIR, exist_ok=True)
    cache = os.path.join(C.DATA_DIR, "prices.csv")
    if synthetic:
        return synthetic_prices()
    if os.path.exists(cache) and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    try:
        import yfinance as yf
        raw = yf.download(list(C.TICKERS.values()), start=C.START, auto_adjust=True, progress=False)["Close"]
    except Exception as e:
        raise RuntimeError(f"Download failed ({e}). Check the internet connection or run with --synthetic.")
    px = raw.rename(columns={v: k for k, v in C.TICKERS.items()})[list(C.TICKERS)]
    px = px.ffill(limit=5).dropna()                        # holidays differ by market: fill small gaps, keep common dates
    px.to_csv(cache)
    return px
