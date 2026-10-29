"""Trading signals in [-1, 1]. They use data up to day t; the backtest trades them on day t+1."""
import numpy as np

import config as C


def momentum_signal(r, lookbacks=(21, 63, 126, 252)):
    """Trend following: past return divided by its normal size (z-score), squashed with tanh, averaged over horizons."""
    d_vol = r.ewm(alpha=1 - C.EWMA_LAMBDA, adjust=False).std()
    return sum(np.tanh(r.rolling(L).sum() / (d_vol * np.sqrt(L)) / 2) for L in lookbacks) / len(lookbacks)


def mean_reversion_signal(r, window=20):
    """Fade stretches: sell when the price is far above its recent average, buy when far below."""
    lp = r.cumsum()
    z = (lp - lp.rolling(window).mean()) / lp.rolling(window).std()
    return -np.tanh(z / 2)
