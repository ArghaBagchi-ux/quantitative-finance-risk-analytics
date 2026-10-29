"""Risk management: VaR / Expected Shortfall, VaR back-test (Kupiec), risk contributions, stress tests, limits."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

import config as C


def var_es_hist(pnl, level=C.VAR_LEVEL):
    q = np.quantile(pnl, 1 - level)
    return -q, -pnl[pnl <= q].mean()


def var_es_normal(pnl, level=C.VAR_LEVEL):
    mu, s, z = pnl.mean(), pnl.std(), stats.norm.ppf(1 - level)
    return -(mu + s * z), -(mu - s * stats.norm.pdf(z) / (1 - level))


def var_es_t(pnl, level=C.VAR_LEVEL):
    df, loc, scale = stats.t.fit(pnl.values)
    x = stats.t.rvs(df, loc, scale, size=400000, random_state=0)
    q = np.quantile(x, 1 - level)
    return -q, -x[x <= q].mean()


def kupiec_pof(x, n, p):
    """Is the number of VaR breaches x out of n days consistent with the expected rate p?"""
    pi = x / n
    lr = -2 * ((xlogy(n - x, 1 - p) + xlogy(x, p)) - (xlogy(n - x, 1 - pi) + xlogy(x, pi)))
    return lr, float(stats.chi2.sf(lr, 1))


def var_backtest(pnl, window=250, level=C.VAR_LEVEL):
    var = -pnl.rolling(window).quantile(1 - level).shift(1)
    ok = var.notna()
    exc = int((pnl[ok] < -var[ok]).sum())
    n = int(ok.sum())
    _, pval = kupiec_pof(exc, n, 1 - level)
    return {"days": n, "breaches": exc, "expected_breaches": round(n * (1 - level), 1), "kupiec_p_value": pval}


def risk_contributions(pos_last, cov_daily):
    w = pos_last.values
    port_vol = np.sqrt(w @ cov_daily.values @ w)
    rc = w * (cov_daily.values @ w) / port_vol
    return pd.DataFrame({"position": w, "risk_share": rc / rc.sum()}, index=pos_last.index), port_vol * np.sqrt(C.TRADING_DAYS)


SCENARIOS = {          # illustrative shocks (assumptions, not forecasts)
    "Equity crash": {"S&P 500 futures": -0.10, "Nifty 50": -0.12, "US 10Y note futures": 0.015, "Gold futures": 0.02,
                     "WTI crude futures": -0.08, "EUR/USD futures": -0.02},
    "Rates shock (+75bp)": {"US 10Y note futures": -0.06, "S&P 500 futures": -0.04, "Gold futures": -0.03, "Nifty 50": -0.03},
    "Oil spike": {"WTI crude futures": 0.25, "S&P 500 futures": -0.03, "Nifty 50": -0.04, "EUR/USD futures": -0.02},
    "USD rally": {"EUR/USD futures": -0.06, "Gold futures": -0.04, "WTI crude futures": -0.04},
}


def stress_test(pos_last):
    return pd.Series({name: sum(pos_last.get(a, 0) * s for a, s in sh.items()) for name, sh in SCENARIOS.items()},
                     name="pnl_pct_of_capital")


@dataclass
class RiskLimits:
    max_pos: float = C.MAX_POS
    max_gross: float = C.MAX_LEVERAGE
    max_port_vol: float = 0.15
    dd_half: float = 0.08
    dd_flat: float = 0.20

    def scale(self, pos, port_vol, drawdown):
        """Return the factor (0..1) to multiply a proposed set of positions by, and the reasons."""
        f, why = 1.0, []
        gross = float(np.abs(pos).sum())
        if gross > self.max_gross:
            f, why = min(f, self.max_gross / gross), why + [f"gross {gross:.2f} > {self.max_gross}"]
        if port_vol > self.max_port_vol:
            f, why = min(f, self.max_port_vol / port_vol), why + [f"portfolio vol {port_vol:.1%} > {self.max_port_vol:.0%}"]
        if drawdown >= self.dd_flat:
            f, why = 0.0, why + [f"drawdown {drawdown:.1%}: flat"]
        elif drawdown >= self.dd_half:
            f, why = min(f, 0.5), why + [f"drawdown {drawdown:.1%}: half size"]
        return f, why
