"""Market analysis: returns, volatility, fat tails, autocorrelation, trend/mean-reversion tests, regimes."""
import numpy as np
import pandas as pd
from scipy import stats

import config as C


def log_returns(px):
    return np.log(px).diff().dropna()


def ewma_vol(r, lam=C.EWMA_LAMBDA):
    """RiskMetrics volatility, annualised."""
    return np.sqrt((r ** 2).ewm(alpha=1 - lam, adjust=False).mean()) * np.sqrt(C.TRADING_DAYS)


def summary_stats(r):
    n = len(r)
    out = pd.DataFrame({
        "ann_return": r.mean() * C.TRADING_DAYS,
        "ann_vol": r.std() * np.sqrt(C.TRADING_DAYS),
        "sharpe": r.mean() / r.std() * np.sqrt(C.TRADING_DAYS),
        "skew": r.skew(), "excess_kurtosis": r.kurt(),
        "worst_day": r.min(), "best_day": r.max(),
        "days_beyond_3sd": ((r - r.mean()).abs() > 3 * r.std()).sum(),
    })
    out["expected_if_normal"] = n * 2 * stats.norm.sf(3)
    out["jarque_bera_p"] = [stats.jarque_bera(r[c]).pvalue for c in r]
    return out


def ljung_box(x, lags=10):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    ac = np.array([np.sum(x[k:] * x[:-k]) / np.sum(x * x) for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(ac ** 2 / (n - np.arange(1, lags + 1)))
    return q, 1 - stats.chi2.cdf(q, lags)


def hurst(logp, max_lag=100):
    """H > 0.5 trending, H < 0.5 mean-reverting, H = 0.5 random walk (log prices)."""
    lags = np.arange(2, max_lag)
    tau = [np.std(logp[l:] - logp[:-l]) for l in lags]
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]


def dependence_table(r, px):
    rows = {}
    n = len(r)
    for c in r:
        ac1 = r[c].autocorr(1)
        rows[c] = {"lag1_autocorr": ac1, "lag1_t_stat": ac1 * np.sqrt(n),
                   "ljung_box_p_returns": ljung_box(r[c].values)[1],
                   "ljung_box_p_squared": ljung_box(r[c].values ** 2)[1],    # volatility clustering
                   "hurst": hurst(np.log(px[c]).values)}
    return pd.DataFrame(rows).T


def regime_table(r):
    """Calm vs stress days (stress = top 20% of cross-asset volatility): does diversification survive stress?"""
    vol = ewma_vol(r).mean(axis=1)
    stress = vol > vol.quantile(0.8)
    rows = {}
    for name, m in (("calm", ~stress), ("stress", stress)):
        sub = r[m]
        c = sub.corr().values
        rows[name] = {"days": int(m.sum()), "avg_ann_vol": sub.std().mean() * np.sqrt(C.TRADING_DAYS),
                      "avg_abs_pairwise_corr": np.abs(c[np.triu_indices_from(c, 1)]).mean()}
    return pd.DataFrame(rows).T
