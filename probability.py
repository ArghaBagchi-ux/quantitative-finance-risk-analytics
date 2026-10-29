"""Probability toolkit: fat tails, Bayesian hit rate, Kelly sizing, Monte Carlo drawdown, luck vs skill."""
import numpy as np
from scipy import stats

from config import TRADING_DAYS


def tail_vs_normal(r, k=4):
    """How often is a move bigger than k standard deviations: observed vs normal model vs fitted Student-t."""
    z = (r - r.mean()) / r.std()
    df, loc, scale = stats.t.fit(r.values)
    return {"k_sigma": k, "observed": float((z.abs() > k).mean()), "normal_model": float(2 * stats.norm.sf(k)),
            "student_t_model": float(2 * stats.t.sf(k * r.std() / scale, df)), "t_degrees_of_freedom": float(df)}


def sharpe(x):
    x = np.asarray(x)
    s = x.std(ddof=1)
    return x.mean() / s * np.sqrt(TRADING_DAYS) if s > 0 else 0.0


def block_bootstrap(x, stat=sharpe, n_boot=2000, block=10, seed=0):
    """Resample blocks of consecutive days (keeps some autocorrelation) and recompute a statistic."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)
    nb = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        out[i] = stat(x[(starts[:, None] + np.arange(block)).ravel()[:n]])
    return out


def bayes_hit_rate(wins, n, prior=(1, 1), level=0.95):
    """Beta posterior of a win probability after seeing `wins` wins in `n` trials."""
    a, b = prior[0] + wins, prior[1] + n - wins
    lo, hi = stats.beta.ppf([(1 - level) / 2, 1 - (1 - level) / 2], a, b)
    return {"mean": a / (a + b), "ci_low": float(lo), "ci_high": float(hi), "p_above_50": float(stats.beta.sf(0.5, a, b))}


def kelly_binary(p, b):
    """Bet fraction that maximises long-run growth: win b per 1 staked with probability p."""
    return max(0.0, (p * b - (1 - p)) / b)


def kelly_continuous(r):
    """Growth-optimal leverage for a return stream (mean / variance). Use half of it in practice."""
    return float(r.mean() / r.var())


def mc_drawdown(mu, sigma, days=TRADING_DAYS, level=0.10, n=20000, df=5, seed=0):
    """Chance of a drawdown of at least `level` within `days`, with fat-tailed (Student-t) daily shocks."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(df, (n, days)) / np.sqrt(df / (df - 2))
    eq = np.concatenate([np.zeros((n, 1)), np.cumsum(mu / TRADING_DAYS + sigma / np.sqrt(TRADING_DAYS) * z, axis=1)], axis=1)
    maxdd = (1 - np.exp(eq - np.maximum.accumulate(eq, axis=1))).max(axis=1)
    return float((maxdd >= level).mean()), maxdd


def prob_sharpe_by_chance(sr, years):
    """If the true Sharpe were 0, the chance of seeing at least `sr` (standard error of Sharpe is about 1/sqrt(years))."""
    return float(stats.norm.sf(sr * np.sqrt(years)))


def years_needed(sr, z=2.0):
    """Years of data needed before a Sharpe of `sr` reaches a t-statistic of z."""
    return float((z / sr) ** 2)
