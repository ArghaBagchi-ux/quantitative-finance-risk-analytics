"""Charts saved as PNG files."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis import ewma_vol

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def prices_and_vol(px, r, path):
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    (px / px.iloc[0] * 100).plot(ax=ax[0], lw=1)
    ax[0].set_title("Prices (rebased to 100)"); ax[0].legend(fontsize=7, ncol=3)
    ewma_vol(r).plot(ax=ax[1], lw=1, legend=False)
    ax[1].set_title("EWMA volatility (annualised)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def correlation(r, path):
    c = r.corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.columns, fontsize=8); ax.grid(False)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im); ax.set_title("Correlation of daily returns")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def equity_curves(curves, path):
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.5]})
    for name, p in curves.items():
        eq = (1 + p).cumprod()
        ax[0].plot(eq, lw=1.2, label=name); ax[1].plot(eq / eq.cummax() - 1, lw=1)
    ax[0].set_title("Growth of 1 unit of capital"); ax[0].legend(fontsize=8)
    ax[1].set_title("Drawdown")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def return_distribution(x, name, path):
    z = (x - x.mean()) / x.std()
    df, loc, scale = stats.t.fit(x.values)
    g = np.linspace(-8, 8, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(z, bins=120, density=True, alpha=0.5, label="observed")
    ax.plot(g, stats.norm.pdf(g), "r", label="normal")
    ax.plot(g, stats.t.pdf(g * x.std(), df, loc - x.mean(), scale) * x.std(), "g", label=f"Student-t (df={df:.1f})")
    ax.set_yscale("log"); ax.set_ylim(1e-5, 1); ax.legend()
    ax.set_title(f"{name}: standardised daily returns (log scale shows the tails)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def hedge_pnl(pnls, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, p in pnls.items():
        ax.hist(p, bins=80, alpha=0.5, label=f"{name} (mean {p.mean():+.2f}, sd {p.std():.2f})")
    ax.axvline(0, color="k", lw=1); ax.legend(fontsize=8)
    ax.set_title("Short call, delta-hedged: P&L at expiry"); ax.set_xlabel("P&L per option")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
