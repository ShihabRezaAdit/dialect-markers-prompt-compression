"""Operating characteristics of the P0 signal rules (pilot plan, Section 8 and Appendix A).
Copied verbatim from the pilot plan's appendix."""
import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(86)
z1 = norm.ppf(1 - (0.05 / 6) / 2)      # 99.17% two-sided, S1 (6 cells)
z2 = norm.ppf(1 - 0.025 / 2)           # 97.5% two-sided, S2 (2 cells)

def p_s1(delta, sigma, n=300, floor=0.04):
    se = sigma / np.sqrt(n); c = max(floor, z1 * se)
    return norm.sf((c - delta) / se) + norm.cdf((-c - delta) / se)

def p_s2(delta, pi, rho=0.3, m=2, n=600, floor=0.03):
    se = np.sqrt((pi - delta**2) * (1 + (m - 1) * rho) / n); c = max(floor, z2 * se)
    return norm.sf((c - delta) / se) + norm.cdf((-c - delta) / se)

def mc_s1(delta, sigma, reps=5000, n=300):
    hits = 0
    for _ in range(reps):
        d = rng.normal(delta, sigma, n).clip(-1, 1)
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
        hits += (abs(m) >= 0.04) and (abs(m) > z1 * se)
    return hits / reps

def mc_s2(delta, pi, rho=0.3, reps=5000, k=300):
    hits = 0; pp, pm = (pi + delta) / 2, (pi - delta) / 2
    for _ in range(reps):
        d = rng.choice([1, -1, 0], size=(k, 2), p=[pp, pm, 1 - pi])
        copy = rng.random(k) < rho; d[copy, 1] = d[copy, 0]
        cl = d.mean(1); m, se = cl.mean(), cl.std(ddof=1) / np.sqrt(k)
        hits += (abs(m) >= 0.03) and (abs(m) > z2 * se)
    return hits / reps

# Draft rule: point estimate of |Delta gamma| >= 0.02 on 200 items
for sd in (0.5, 0.7):
    print(sd, 2 * norm.sf(0.02 / (sd / np.sqrt(200))))
