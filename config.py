"""Settings for the whole project. Edit here."""
TICKERS = {                      # name -> Yahoo Finance symbol (a mix of equity, rates, commodity, FX)
    "S&P 500 futures": "ES=F",
    "US 10Y note futures": "ZN=F",
    "WTI crude futures": "CL=F",
    "Gold futures": "GC=F",
    "EUR/USD futures": "6E=F",
    "Nifty 50": "^NSEI",
}
START = "2012-01-01"
TRADING_DAYS = 252
TARGET_VOL = 0.10        # annual volatility target for the whole portfolio
COST_BPS = 2.0           # cost per unit of turnover, in basis points
MAX_LEVERAGE = 3.0       # cap on gross exposure (sum of |position|, as a multiple of capital)
MAX_POS = 1.0            # cap on one asset (as a multiple of capital)
VAR_LEVEL = 0.99
EWMA_LAMBDA = 0.94       # RiskMetrics decay
WARMUP = 253             # days skipped at the start (longest momentum look-back + 1)
DATA_DIR, RESULTS_DIR = "data", "results"
SEED = 42
