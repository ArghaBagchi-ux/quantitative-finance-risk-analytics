"""Fast, rule-based trade decisions with a statistical gate, volatility sizing, stops and a latency log.

Rules (in order):
 1. Score = 0.7 x momentum + 0.3 x mean-reversion signal.
 2. Statistical gate: look at past days with a similar-strength signal and count how often its direction was right.
    Take the lower end of a Bayesian credible interval for that hit rate. If it is not above 50%, do nothing.
 3. Size = |score| x risk budget / volatility, scaled up to full size once the hit-rate lower bound is 5 points above 50%.
 4. Halve the size when current volatility is more than twice its median (stress regime).
 5. Stop = 2 daily sigmas, target = 3 daily sigmas (reward/risk 1.5).
"""
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

import config as C
from analysis import ewma_vol
from probability import bayes_hit_rate
from signals import mean_reversion_signal, momentum_signal


@dataclass
class Decision:
    asset: str
    action: str
    size: float
    hit_rate_lower: float
    stop: float
    target: float
    reason: str
    latency_ms: float


class DecisionEngine:
    def __init__(self, window=1000, min_signal=0.20, risk_budget=0.04):
        self.window, self.min_signal, self.risk_budget = window, min_signal, risk_budget

    def decide(self, asset, rets, price):
        t0 = time.perf_counter()
        r = rets.iloc[-self.window:].to_frame("x")
        score = (0.7 * momentum_signal(r) + 0.3 * mean_reversion_signal(r))["x"]
        s, lagged = score.iloc[-1], score.shift(1)
        mask = lagged.abs() > self.min_signal
        n = int(mask.sum())
        hits = int((np.sign(lagged[mask]) == np.sign(r["x"][mask])).sum())
        lo = bayes_hit_rate(hits, n, level=0.6)["ci_low"] if n else 0.5       # lower 20% point
        vol_series = ewma_vol(r)["x"]
        vol, dvol = vol_series.iloc[-1], vol_series.iloc[-1] / np.sqrt(C.TRADING_DAYS)
        action, size, stop, target = "HOLD", 0.0, np.nan, np.nan
        if n < 30:
            why = f"only {n} comparable past signals: not enough evidence"
        elif lo <= 0.5:
            why = f"hit-rate lower bound {lo:.1%} on {n} past signals is not above 50%: no proven edge"
        elif abs(s) < self.min_signal:
            why = f"signal {s:+.2f} too weak"
        else:
            action = "BUY" if s > 0 else "SELL"
            size = min(C.MAX_POS, abs(s) * self.risk_budget / vol) * min(1.0, (lo - 0.5) / 0.05)
            why = f"score {s:+.2f}, hit-rate lower bound {lo:.1%} on {n} signals"
            if vol > 2 * vol_series.median():
                size, why = size / 2, why + "; high-volatility regime: size halved"
            sgn = 1 if action == "BUY" else -1
            stop, target = price * (1 - sgn * 2 * dvol), price * (1 + sgn * 3 * dvol)
        return Decision(asset, action, round(float(size), 3), round(float(lo), 3), round(float(stop), 2),
                        round(float(target), 2), why, round((time.perf_counter() - t0) * 1000, 2))

    def decide_all(self, px, r):
        return pd.DataFrame([asdict(self.decide(a, r[a], px[a].iloc[-1])) for a in r.columns]).set_index("asset")

    def latency_benchmark(self, px, r, n=100):
        """Time n decisions on different past days (what a live desk would need: milliseconds, not seconds)."""
        ms = []
        for i in np.linspace(len(r) - 700, len(r) - 1, n).astype(int):
            a = r.columns[i % r.shape[1]]
            ms.append(self.decide(a, r[a].iloc[:i + 1], px[a].iloc[i]).latency_ms)
        return {"decisions": n, "median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)), "max_ms": float(max(ms))}
