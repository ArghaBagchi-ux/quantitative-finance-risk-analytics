"""Volatility-targeted backtest with costs, performance metrics, drawdown circuit breaker and walk-forward testing."""
import numpy as np
import pandas as pd

import config as C
from analysis import ewma_vol
from probability import block_bootstrap, sharpe


def vol_target_positions(sig, r, target_vol=C.TARGET_VOL):
    """Position (fraction of capital) per asset: signal x risk budget / volatility, then caps."""
    raw = sig * (target_vol / np.sqrt(r.shape[1])) / ewma_vol(r)
    pos = raw.clip(-C.MAX_POS, C.MAX_POS)
    gross = pos.abs().sum(axis=1).replace(0, np.nan)
    return pos.mul(np.minimum(1, C.MAX_LEVERAGE / gross).fillna(1), axis=0)


def run_backtest(r, sig, cost_bps=C.COST_BPS, breaker=False):
    """Positions chosen at the close of day t-1 earn the return of day t. Costs are charged on turnover."""
    target = vol_target_positions(sig, r).shift(1).fillna(0)
    if breaker:                                         # cut risk after losses: <8% dd full, 8-15% half, >15% quarter
        eq = peak = 1.0
        prev = np.zeros(r.shape[1])
        pos_arr, pnl_arr = np.zeros(target.shape), np.zeros(len(r))
        for t in range(len(r)):
            dd = 1 - eq / peak
            p = target.values[t] * (1.0 if dd < 0.08 else 0.5 if dd < 0.15 else 0.25)
            pnl_arr[t] = (p * r.values[t]).sum() - cost_bps * 1e-4 * np.abs(p - prev).sum()
            eq *= 1 + pnl_arr[t]
            peak = max(peak, eq)
            pos_arr[t], prev = p, p
        pos = pd.DataFrame(pos_arr, index=r.index, columns=r.columns)
        pnl = pd.Series(pnl_arr, index=r.index)
    else:
        pos = target
        pnl = (pos * r).sum(axis=1) - cost_bps * 1e-4 * pos.diff().abs().sum(axis=1).fillna(0)
    return {"pnl": pnl.iloc[C.WARMUP:], "pos": pos.iloc[C.WARMUP:]}


def metrics(pnl):
    eq = (1 + pnl).cumprod()
    dd = eq / eq.cummax() - 1
    ann, vol = pnl.mean() * C.TRADING_DAYS, pnl.std() * np.sqrt(C.TRADING_DAYS)
    down = pnl[pnl < 0].std() * np.sqrt(C.TRADING_DAYS)
    return {"ann_return": ann, "ann_vol": vol, "sharpe": ann / vol, "sortino": ann / down,
            "max_drawdown": dd.min(), "calmar": ann / abs(dd.min()), "hit_rate": (pnl > 0).mean(),
            "profit_factor": pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum()),
            "t_stat": pnl.mean() / pnl.std() * np.sqrt(len(pnl))}


def sharpe_ci(pnl, level=0.95):
    b = block_bootstrap(pnl.values, sharpe)
    return tuple(np.quantile(b, [(1 - level) / 2, 1 - (1 - level) / 2]))


def walk_forward(pnls, train=3 * C.TRADING_DAYS, test=C.TRADING_DAYS):
    """pnls: {parameter: pnl series}. On each training window pick the parameter with the best Sharpe,
    then use it on the next unseen window. This is the honest estimate: no peeking at the test data."""
    idx = next(iter(pnls.values())).index
    oos, rows = [], []
    for s in range(train, len(idx), test):
        tr, te = idx[s - train:s], idx[s:s + test]
        sh = {w: np.nan_to_num(p.loc[tr].mean() / p.loc[tr].std(), nan=-9) for w, p in pnls.items()}
        w = max(sh, key=sh.get)
        oos.append(pnls[w].loc[te])
        rows.append({"test_start": str(te[0].date()), "chosen_mom_weight": w, "in_sample_sharpe": sh[w] * np.sqrt(C.TRADING_DAYS),
                     "out_of_sample_sharpe": pnls[w].loc[te].mean() / pnls[w].loc[te].std() * np.sqrt(C.TRADING_DAYS)})
    return pd.concat(oos), pd.DataFrame(rows)
