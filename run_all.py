"""
Runs the whole project end to end and writes results/report.md plus charts.

    python run_all.py                 # real data (downloads from Yahoo Finance the first time)
    python run_all.py --synthetic     # offline test with fake data (results are NOT meaningful)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import config as C
import options as op
import plots
from analysis import dependence_table, log_returns, regime_table, summary_stats
from backtest import metrics, run_backtest, sharpe_ci, walk_forward
from data import load_prices
from decision_engine import DecisionEngine
from probability import (bayes_hit_rate, kelly_continuous, mc_drawdown, prob_sharpe_by_chance, tail_vs_normal, years_needed)
from risk import RiskLimits, risk_contributions, stress_test, var_backtest, var_es_hist, var_es_normal, var_es_t
from signals import mean_reversion_signal, momentum_signal


def md(df, fmt="{:.3f}"):
    head = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, row in df.iterrows():
        cells = [str(int(v)) if isinstance(v, (float, np.floating)) and float(v).is_integer() else fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join([str(i)] + cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default=C.RESULTS_DIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rep = []
    add = rep.append

    px = load_prices(a.synthetic, a.refresh)
    r = log_returns(px)
    print(f"Data: {px.shape[1]} assets, {len(r)} days, {r.index[0].date()} to {r.index[-1].date()}")
    add("# Futures-style markets project: analysis, probability, risk and decisions\n")
    if a.synthetic:
        add("> **SYNTHETIC DATA.** Generated for testing the code offline. None of the numbers below describe real markets.\n")
    add(f"Data: {px.shape[1]} assets, {len(r)} daily returns, {r.index[0].date()} to {r.index[-1].date()}.\n")

    # 1 ---- market analysis
    add("## 1. Market analysis\n")
    ss = summary_stats(r)
    add(md(ss.round(4)) + "\n")
    add("`days_beyond_3sd` far above `expected_if_normal` means fat tails: extreme days happen much more often than a normal model says.\n")
    add("### Memory, trend and volatility clustering\n")
    add(md(dependence_table(r, px)) + "\n")
    add("Hurst above 0.5 suggests trending, below 0.5 mean-reverting. A small `ljung_box_p_squared` means volatility clusters (calm and wild periods).\n")
    add("### Calm vs stress days\n")
    add(md(regime_table(r)) + "\n")
    add("Correlations usually rise in stress, so diversification protects least when it is needed most.\n")
    tails = pd.DataFrame({c: tail_vs_normal(r[c]) for c in r}).T
    add("### 4-sigma days: observed vs models\n")
    add(md(tails, "{:.5f}") + "\n")
    plots.prices_and_vol(px, r, f"{a.out}/01_prices_vol.png")
    plots.correlation(r, f"{a.out}/02_correlation.png")
    plots.return_distribution(r.iloc[:, 0], r.columns[0], f"{a.out}/03_tails.png")
    add("![prices](01_prices_vol.png)\n![correlation](02_correlation.png)\n![tails](03_tails.png)\n")

    # 2 ---- signals and backtest
    add("## 2. Strategy research (vol-targeted, after costs)\n")
    mom, mr = momentum_signal(r), mean_reversion_signal(r)
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    pnls = {w: run_backtest(r, w * mom + (1 - w) * mr)["pnl"] for w in grid}
    blend = run_backtest(r, 0.5 * mom + 0.5 * mr)
    guarded = run_backtest(r, 0.5 * mom + 0.5 * mr, breaker=True)
    oos, folds = walk_forward(pnls)
    table = pd.DataFrame({"Mean-reversion only": metrics(pnls[0.0]), "Momentum only": metrics(pnls[1.0]),
                          "50/50 blend": metrics(blend["pnl"]), "50/50 + drawdown breaker": metrics(guarded["pnl"]),
                          "Walk-forward (out of sample)": metrics(oos)}).T
    add(md(table) + "\n")
    add(f"Volatility is below the {C.TARGET_VOL:.0%} target because signals are rarely at full strength (positions scale with signal size).\n")
    lo, hi = sharpe_ci(oos)
    add(f"Out-of-sample Sharpe {table.loc['Walk-forward (out of sample)', 'sharpe']:.2f}, 95% bootstrap interval [{lo:.2f}, {hi:.2f}]. "
        "If the interval includes 0, the edge is not statistically proven.\n")
    add("### Walk-forward folds (parameter chosen on 3 years, tested on the next unseen year)\n")
    add(md(folds.set_index("test_start")) + "\n")
    sig = 0.7 * mom + 0.3 * mr
    lag = sig.shift(1)
    mask = lag.abs() > 0.3
    n_sig = int(mask.sum().sum())
    hits = int(((np.sign(lag) == np.sign(r)) & mask).sum().sum())
    post = bayes_hit_rate(hits, n_sig)
    add(f"### Is the direction call better than a coin flip?\n{hits} of {n_sig} strong signals were right ({hits / n_sig:.1%}). "
        f"Bayesian 95% interval for the true hit rate: [{post['ci_low']:.1%}, {post['ci_high']:.1%}]; "
        f"probability it is above 50%: {post['p_above_50']:.1%}.\n")
    plots.equity_curves({"Momentum": pnls[1.0], "Mean-reversion": pnls[0.0], "50/50 blend": blend["pnl"],
                         "Blend + breaker": guarded["pnl"], "Walk-forward OOS": oos}, f"{a.out}/04_equity.png")
    add("![equity](04_equity.png)\n")

    # 3 ---- probability
    add("## 3. Probability toolkit\n")
    pn = blend["pnl"]
    sr, yrs = table.loc["50/50 blend", "sharpe"], len(pn) / C.TRADING_DAYS
    add(f"- **Luck vs skill.** Sharpe {sr:.2f} over {yrs:.1f} years: chance of seeing at least this if the true Sharpe were 0 = "
        f"{prob_sharpe_by_chance(sr, yrs):.1%}. " + (f"A true Sharpe of {sr:.2f} needs about {years_needed(sr):.0f} years of data to reach t = 2.\n" if sr > 0 else "\n"))
    p_dd, maxdd = mc_drawdown(pn.mean() * C.TRADING_DAYS, pn.std() * np.sqrt(C.TRADING_DAYS), level=0.10)
    add(f"- **Monte Carlo drawdown.** With the blend's return and volatility and fat-tailed shocks, the chance of a drawdown of at least 10% "
        f"within a year is {p_dd:.1%} (median worst drawdown {np.median(maxdd):.1%}, 95th percentile {np.quantile(maxdd, 0.95):.1%}).\n")
    kel = r.apply(kelly_continuous)
    add("- **Kelly leverage** (mean / variance of daily returns, per asset): " + ", ".join(f"{k} {v:.1f}x" for k, v in kel.items())
        + ". Estimated means are noisy, so these are far too high to use; professionals trade a fraction (half Kelly or less).\n")

    # 4 ---- risk
    add("## 4. Risk management\n")
    rows = {}
    for lvl in (0.95, 0.99):
        for name, f in (("historical", var_es_hist), ("normal", var_es_normal), ("Student-t", var_es_t)):
            v, e = f(pn, lvl)
            rows[f"{name} {lvl:.0%}"] = {"VaR (1 day)": v, "Expected shortfall": e}
    add(md(pd.DataFrame(rows).T, "{:.4%}") + "\n")
    vb = var_backtest(pn)
    add(f"VaR back-test (rolling 250-day historical, 99%): {vb['breaches']} breaches in {vb['days']} days, expected about {vb['expected_breaches']}; "
        f"Kupiec p-value {vb['kupiec_p_value']:.3f} (small values mean the VaR model is mis-calibrated).\n")
    last = blend["pos"].iloc[-1]
    rc, pvol = risk_contributions(last, r.tail(250).cov())
    add(f"### Latest positions and risk contributions (portfolio volatility {pvol:.1%})\n" + md(rc, "{:.3f}") + "\n")
    add("### Stress scenarios (illustrative shocks), P&L as % of capital\n" + md(stress_test(last).to_frame(), "{:.2%}") + "\n")
    eq = (1 + pn).cumprod()
    f, why = RiskLimits().scale(last, pvol, float(1 - eq.iloc[-1] / eq.max()))
    add(f"Risk-limit check on the latest book: scale factor {f:.2f}" + (" (" + "; ".join(why) + ")" if why else " (all limits respected)") + ".\n")

    # 5 ---- derivatives
    add("## 5. Derivatives: pricing, Greeks, hedging\n")
    S, K, T, rf, sg = 100, 100, 0.25, 0.05, 0.20
    px_c = op.bs_price(S, K, T, rf, sg)
    g = op.greeks(S, K, T, rf, sg)
    add(f"At-the-money 3-month call (S=K=100, r=5%, vol 20%): price {px_c:.3f}; implied vol recovered from that price: {op.implied_vol(px_c, S, K, T, rf):.4f}.\n")
    add(md(pd.DataFrame(g, index=["value"]).T, "{:.4f}") + "\n")
    hed = {}
    for real in (0.15, 0.20, 0.25):
        for steps in (5, 63):
            p = op.delta_hedge_pnl(sig_imp=0.20, sig_real=real, steps=steps)
            hed[f"realised vol {real:.0%}, {steps} hedges"] = {"mean P&L": p.mean(), "std dev": p.std(),
                                                              "theory (price diff)": op.bs_price(100, 100, 0.25, 0, 0.20) - op.bs_price(100, 100, 0.25, 0, real)}
    add("Sell a call at 20% implied vol and delta-hedge. If realised vol is lower you earn, if higher you lose; hedging more often cuts the noise:\n")
    add(md(pd.DataFrame(hed).T) + "\n")
    plots.hedge_pnl({f"{s} hedges": op.delta_hedge_pnl(sig_real=0.25, steps=s) for s in (5, 63)}, f"{a.out}/05_hedging.png")
    add("![hedging](05_hedging.png)\n")

    # 6 ---- decisions
    add("## 6. Decision engine (latest day)\n")
    eng = DecisionEngine()
    dec = eng.decide_all(px, r)
    add(md(dec.drop(columns=["latency_ms"]).astype(object).fillna("-"), "{:.3f}") + "\n")
    lat = eng.latency_benchmark(px, r)
    add(f"Speed: {lat['decisions']} decisions, median {lat['median_ms']:.1f} ms, 95th percentile {lat['p95_ms']:.1f} ms, slowest {lat['max_ms']:.1f} ms.\n")

    add("## 7. Limitations\n")
    add("- Daily data and simple rules; no intraday data, bid/ask spreads, slippage, margin or contract roll effects beyond a flat cost.\n"
        "- Yahoo continuous futures are back-adjusted and can contain gaps; check data quality before trusting results.\n"
        "- Past performance does not predict future returns. Stress shocks are assumptions, not forecasts.\n"
        "- This is a learning and portfolio project, not investment advice.\n")
    with open(f"{a.out}/report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(rep))
    table.to_csv(f"{a.out}/strategy_metrics.csv"); folds.to_csv(f"{a.out}/walk_forward_folds.csv", index=False); ss.to_csv(f"{a.out}/summary_stats.csv")
    print(f"Done. Open {a.out}/report.md")


if __name__ == "__main__":
    main()
