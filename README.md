# Futures-style markets project: analysis, probability, risk and fast decisions

An end-to-end project that covers what a derivatives trading trainee role asks for:
market analysis, quantitative reasoning, probability, statistics, risk management, fast decision-making.

## Run it
    pip install -r requirements.txt
    python run_all.py                  # real data (Yahoo Finance download the first time, cached in data/)
    python run_all.py --synthetic      # offline test with fake data (numbers are NOT meaningful)
    python drills.py --n 10 --time 30  # timed quant questions in the terminal
Open `results/report.md` (charts and CSV tables are saved next to it). `demo_results_synthetic/` shows what the output looks like.

## What each skill maps to
| Skill | Where | What it shows |
|---|---|---|
| Market analysis | `analysis.py`, section 1 | returns, EWMA volatility, fat tails, autocorrelation, Hurst exponent, volatility clustering, calm vs stress correlation |
| Statistics | `analysis.py`, `backtest.py`, `probability.py` | Jarque-Bera, Ljung-Box, t-statistics, block-bootstrap confidence interval for Sharpe, walk-forward testing |
| Probability | `probability.py`, section 3 | Student-t tails, Bayesian hit rate (Beta), Kelly sizing, Monte Carlo drawdown odds, luck vs skill |
| Quantitative reasoning | `drills.py`, `options.py` | timed drills (mental maths, EV, Bayes, Kelly, futures P&L); Black-Scholes, Greeks, implied vol, delta-hedging experiment |
| Risk management | `risk.py`, `backtest.py`, section 4 | VaR and expected shortfall (3 methods), Kupiec back-test, risk contributions, stress scenarios, limits, drawdown circuit breaker |
| Fast decisions | `decision_engine.py`, section 6 | rule-based BUY/SELL/HOLD with a statistical gate, vol sizing, stop and target, latency in milliseconds |

## Files
`config.py` settings | `data.py` data | `analysis.py` | `signals.py` momentum and mean-reversion | `backtest.py` | `probability.py` | `risk.py` |
`options.py` | `decision_engine.py` | `drills.py` | `plots.py` | `run_all.py` runs everything.


* **No look-ahead.** Signals use data up to day t and trade on day t+1; costs are charged on turnover.
* **Volatility targeting.** Position = signal x risk budget / volatility, with per-asset and gross-leverage caps.
* **Honest testing.** Walk-forward: choose the parameter on 3 years, test on the next unseen year. Report a bootstrap interval, not just one Sharpe.
* **Decision gate.** The engine trades only if the lower end of a Bayesian interval for the past hit rate is above 50%, otherwise it holds.
* **Fat tails.** A normal model badly underestimates extreme days; VaR is shown with historical, normal and Student-t methods.
* **Risk first.** Circuit breaker (half size at 8% drawdown, quarter at 15%), limit checks and stress tests sit beside the P&L.

## Limits
Daily data, flat costs, no slippage, margin or contract-roll modelling. Yahoo continuous futures can contain gaps.
Stress shocks are assumptions. This is a learning and portfolio project, not investment advice.

## Ideas to extend it
Use a volatility index (e.g. VIX) to compare implied and realised volatility; add intraday data; add a GARCH model;
add a pairs-trading or spread strategy; compare more parameter grids with a multiple-testing correction.
