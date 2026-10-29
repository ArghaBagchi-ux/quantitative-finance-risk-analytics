"""Timed quant drills: mental maths, probability, expected value, Bayes, Kelly, futures P&L, portfolio risk.

    python drills.py --n 10 --time 30          # 10 random questions, 30 seconds each
Answers are numbers (decimals, fractions like 1/6, or with % / $). A log is saved to results/drill_log.csv.
"""
import argparse
import csv
import math
import os
import random
import time

from scipy.stats import binom


def q_mult(g):
    a, b = g.randint(12, 98), g.randint(12, 98)
    return dict(topic="mental maths", prompt=f"{a} x {b} = ?", answer=a * b, tol=0)


def q_pct(g):
    p0 = g.randint(80, 400)
    p1 = p0 + g.choice([-1, 1]) * g.randint(3, p0 // 4)
    return dict(topic="mental maths", prompt=f"A price moves from {p0} to {p1}. Percentage change (1 decimal)?",
                answer=round((p1 - p0) / p0 * 100, 2), tol=0.1)


def q_dice(g):
    k = g.randint(3, 11)
    return dict(topic="probability", prompt=f"Two fair dice. Probability that the sum equals {k}? (decimal)",
                answer=(6 - abs(k - 7)) / 36, tol=0.005)


def q_ev(g):
    x, w = g.choice([5, 8, 10, 12]), g.choice([4, 5, 6, 10])
    return dict(topic="expected value",
                prompt=f"A fair die is rolled. If the face is even you win ${w} x face, otherwise nothing. The game costs ${x}. Expected profit?",
                answer=2 * w - x, tol=0.01)


def q_bayes(g):
    prev, sens, fp = g.choice([0.01, 0.02, 0.05]), g.choice([0.9, 0.95]), g.choice([0.05, 0.1])
    return dict(topic="Bayes", prompt=f"A test finds {sens:.0%} of true cases and wrongly flags {fp:.0%} of healthy people. "
                f"Prevalence is {prev:.0%}. Probability of the condition given a positive test? (decimal)",
                answer=sens * prev / (sens * prev + fp * (1 - prev)), tol=0.005)


def q_binom(g):
    k = g.choice([7, 8, 9])
    return dict(topic="probability", prompt=f"10 fair coin flips. Probability of at least {k} heads? (decimal)",
                answer=float(binom.sf(k - 1, 10, 0.5)), tol=0.005)


def q_futures(g):
    e, d, c = g.randint(4800, 6200), g.randint(-60, 60), g.randint(1, 5)
    return dict(topic="futures P&L", prompt=f"You buy {c} E-mini S&P 500 contracts at {e} ($50 per index point) and sell at {e + d}. P&L in $?",
                answer=d * 50 * c, tol=1)


def q_kelly(g):
    p, b = g.choice([0.55, 0.6, 0.65]), g.choice([1, 1.5, 2])
    return dict(topic="Kelly", prompt=f"A bet wins {b} per 1 staked with probability {p}. Kelly fraction of capital to stake? (decimal)",
                answer=(p * b - (1 - p)) / b, tol=0.005)


def q_portvol(g):
    s1, s2, rho = g.choice([10, 15, 20]), g.choice([10, 15, 25]), g.choice([0, 0.5, -0.5, 0.3])
    return dict(topic="portfolio risk", prompt=f"50/50 portfolio of two assets with volatilities {s1}% and {s2}% and correlation {rho}. Portfolio volatility in %?",
                answer=math.sqrt(0.25 * s1 ** 2 + 0.25 * s2 ** 2 + 0.5 * rho * s1 * s2), tol=0.1)


def q_option(g):
    K, c, S = g.choice([100, 105, 110]), g.choice([2, 3, 4]), g.randint(90, 125)
    return dict(topic="options", prompt=f"You buy a call, strike {K}, premium {c}. The underlying ends at {S}. Profit per unit?",
                answer=max(S - K, 0) - c, tol=0.01)


GENERATORS = [q_mult, q_pct, q_dice, q_ev, q_bayes, q_binom, q_futures, q_kelly, q_portvol, q_option]


def make_questions(n=10, seed=None):
    g = random.Random(seed)
    order = (GENERATORS * (n // len(GENERATORS) + 1))
    g.shuffle(order)
    return [f(g) for f in order[:n]]


def parse(text):
    t = text.strip().replace("$", "").replace("%", "").replace(",", "")
    if "/" in t:
        a, b = t.split("/")
        return float(a) / float(b)
    return float(t)


def run(n, limit, seed=None):
    os.makedirs("results", exist_ok=True)
    rows = []
    for i, q in enumerate(make_questions(n, seed), 1):
        print(f"\nQ{i} [{q['topic']}]  ({limit}s)\n{q['prompt']}")
        t0 = time.time()
        try:
            ans = parse(input("> "))
        except ValueError:
            ans = float("nan")
        dt = time.time() - t0
        ok = abs(ans - q["answer"]) <= q["tol"] + 1e-9
        late = dt > limit
        print(("correct" if ok else f"wrong, answer = {q['answer']:.4g}") + (" (too slow)" if late else "") + f"   {dt:.1f}s")
        rows.append({"topic": q["topic"], "correct": int(ok), "seconds": round(dt, 1), "late": int(late)})
    acc, avg = sum(r["correct"] for r in rows) / n, sum(r["seconds"] for r in rows) / n
    print(f"\nScore {sum(r['correct'] for r in rows)}/{n} ({acc:.0%}), average time {avg:.1f}s, late answers {sum(r['late'] for r in rows)}")
    with open("results/drill_log.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if f.tell() == 0:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--time", type=int, default=30, help="seconds allowed per question")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    run(a.n, a.time, a.seed)
