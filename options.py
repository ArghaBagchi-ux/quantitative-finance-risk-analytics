"""Derivatives: Black-Scholes price and Greeks, implied volatility, and a delta-hedging simulation."""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d(S, K, T, r, sig):
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    return d1, d1 - sig * np.sqrt(T)


def bs_price(S, K, T, r, sig, cp="call"):
    d1, d2 = _d(S, K, T, r, sig)
    if cp == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sig, cp="call"):
    d1, _ = _d(S, K, T, r, sig)
    return norm.cdf(d1) if cp == "call" else norm.cdf(d1) - 1


def greeks(S, K, T, r, sig, cp="call"):
    d1, d2 = _d(S, K, T, r, sig)
    theta = -S * norm.pdf(d1) * sig / (2 * np.sqrt(T))
    theta += (-r * K * np.exp(-r * T) * norm.cdf(d2)) if cp == "call" else (r * K * np.exp(-r * T) * norm.cdf(-d2))
    return {"delta": float(bs_delta(S, K, T, r, sig, cp)), "gamma": float(norm.pdf(d1) / (S * sig * np.sqrt(T))),
            "vega_per_vol_pt": float(S * norm.pdf(d1) * np.sqrt(T) / 100), "theta_per_day": float(theta / 365),
            "rho_per_1pct": float((K * T * np.exp(-r * T) * (norm.cdf(d2) if cp == "call" else -norm.cdf(-d2))) / 100)}


def implied_vol(price, S, K, T, r, cp="call"):
    return brentq(lambda s: bs_price(S, K, T, r, s, cp) - price, 1e-4, 5.0)


def delta_hedge_pnl(S0=100.0, K=100.0, T=0.25, r=0.0, sig_imp=0.20, sig_real=0.25, steps=63, n=5000, seed=0):
    """Sell a call at implied vol, hedge delta each step, realised vol may differ. Returns P&L per path at expiry.
    In theory the average P&L is about price(sig_imp) - price(sig_real): you are short volatility."""
    rng = np.random.default_rng(seed)
    dt = T / steps
    S = np.full(n, S0)
    delta = bs_delta(S, K, T, r, sig_imp)
    cash = bs_price(S0, K, T, r, sig_imp) - delta * S
    for i in range(1, steps + 1):
        S = S * np.exp((r - 0.5 * sig_real ** 2) * dt + sig_real * np.sqrt(dt) * rng.standard_normal(n))
        cash = cash * np.exp(r * dt)
        if i < steps:
            new = bs_delta(S, K, T - i * dt, r, sig_imp)
            cash -= (new - delta) * S
            delta = new
    return cash + delta * S - np.maximum(S - K, 0)
