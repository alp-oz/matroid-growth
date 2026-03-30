"""
Negative binomial fit to N_s (circuits of size s) across 50 replicates.

γ=0, λ=0.05, α=0.3, β=0.5, k_t=floor(0.3·r_t), n_steps=2000, start_r=10.
s ∈ {3, 5, 8, 10, 12, 15}.

For each s:
  1. Re-run the 50 replicates (same seeds as pa_circuit_sizes.py).
  2. Fit NB(r, p) by MLE: parameterised by real r > 0, p ∈ (0,1).
       log P(X=k) = logΓ(k+r) − logΓ(r) − logΓ(k+1) + r·log p + k·log(1−p)
     MLE via profile log-likelihood: μ̂ = x̄, then optimise over log r.
     p̂ = r̂/(r̂ + μ̂).
  3. Bootstrap 95% CI on r̂ (1000 bootstrap resamples of the 50 replicates).
  4. Chi-squared goodness-of-fit: 8 equal-frequency bins, merge until
     expected ≥ 5, report χ², df, p-value (ddof=2 for r and p).
  5. KS statistic D = sup_x |F_n(x) − F_{NB}(x)| (informational; p-value
     is approximate for discrete distributions).

Theoretical motivation: the PA clustering creates a Poisson mixture.
Hub rows attract many columns → each hub contributes ≈ Poisson(λ_hub)
circuits → sum of independent Poisson mixtures ~ Negative Binomial.
Small r̂ ≈ 2–13 = effective number of independent hubs driving N_s.

Saves: circuit_nb_fit.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import gammaln, betainc, digamma
from scipy.optimize import minimize_scalar
from scipy.stats import chisquare
import os

ALPHA   = 0.3
BETA    = 0.5
LAMBDA  = 0.05
GAMMA   = 0.0
N_STEPS = 2000
START_R = 10
REPS    = 50
SIZES   = [3, 5, 8, 10, 12, 15]
N_BOOT  = 1000
OUT     = os.path.dirname(__file__)


# ── Simulation (identical to pa_circuit_sizes.py) ────────────────────────────

def run_and_collect(n_steps, alpha, C_lam, gamma, beta, start_r, seed):
    rng       = np.random.default_rng(seed)
    curr_r    = start_r
    row_usage = np.ones(start_r, dtype=np.float64)
    supports, k_list = [], []
    for t in range(1, n_steps + 1):
        p_row = min(1.0, C_lam * t ** (-gamma))
        if rng.random() < p_row:
            curr_r += 1
            row_usage = np.append(row_usage, 1.0)
        else:
            k   = max(1, int(np.floor(alpha * curr_r)))
            k   = min(k, curr_r)
            w   = row_usage ** beta;  w /= w.sum()
            sel = rng.choice(curr_r, size=k, replace=False, p=w)
            supports.append(sel);  k_list.append(k)
            for idx in sel:
                row_usage[int(idx)] += 1
    return supports, np.array(k_list, dtype=np.int32), curr_r


def count_by_size(supports, k_vals, r_final, target_sizes):
    n_att  = len(supports)
    result = {s: 0 for s in target_sizes}
    for s in target_sizes:
        result[s] += int(np.sum(k_vals == s - 1))
    if n_att == 0:
        return result
    B = np.zeros((n_att, r_final), dtype=np.float32)
    for j, sel in enumerate(supports):
        B[j, sel] = 1.0
    G   = (B @ B.T).round().astype(np.int32)
    k32 = k_vals.astype(np.int32)
    D   = k32[:, None] + k32[None, :] - 2 * G
    tri = np.triu(np.ones((n_att, n_att), dtype=bool), k=1)
    overlap = (G > 0) & tri
    for s in target_sizes:
        result[s] += int(np.sum((D == s - 2) & overlap))
    return result


# ── NB utilities ──────────────────────────────────────────────────────────────

def nb_logpmf(k, r, p):
    """Vectorised log P(X=k) for NB(r, p)."""
    k = np.asarray(k, dtype=float)
    return (gammaln(k + r) - gammaln(r) - gammaln(k + 1)
            + r * np.log(p) + k * np.log(1.0 - p))

def nb_cdf(k, r, p):
    """P(X ≤ k) via regularised incomplete beta I_p(r, k+1)."""
    k = float(k)
    if k < 0:
        return 0.0
    return float(betainc(r, k + 1, p))

def fit_nb(data):
    """MLE for NB(r, p). Returns (r_hat, p_hat, converged)."""
    data = np.asarray(data, dtype=float)
    mu   = data.mean()
    if mu <= 0:
        return float("nan"), float("nan"), False
    def neg_ll(log_r):
        r = np.exp(log_r)
        p = r / (r + mu)
        return -np.sum(nb_logpmf(data, r, p))
    res   = minimize_scalar(neg_ll, bounds=(-5, 16), method="bounded",
                            options={"xatol": 1e-8})
    r_hat = np.exp(res.x)
    p_hat = r_hat / (r_hat + mu)
    return float(r_hat), float(p_hat), res.success

def bootstrap_r_ci(data, n_boot=1000, rng=None, alpha=0.05):
    """Bootstrap 95% CI for r̂."""
    rng  = np.random.default_rng(rng)
    data = np.asarray(data)
    rs   = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        r, _, ok = fit_nb(sample)
        if ok and np.isfinite(r):
            rs.append(r)
    rs = np.array(rs)
    lo = float(np.percentile(rs, 100 * alpha / 2))
    hi = float(np.percentile(rs, 100 * (1 - alpha / 2)))
    return lo, hi

def chi2_gof(data, r, p, n_bins=8, ddof=2):
    """
    Chi-squared GoF with quantile-based bins, merged so expected ≥ 5.
    Returns (chi2_stat, p_value, df, n_bins_used).
    """
    from scipy.stats import chisquare as _cs
    n    = len(data)
    data = np.sort(np.asarray(data, dtype=float))
    q    = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(data, q)).astype(float)
    if len(edges) < 3:
        return float("nan"), float("nan"), 0, 0

    raw_obs, raw_exp = [], []
    for i in range(len(edges) - 1):
        a = int(edges[i]);  b = int(edges[i + 1])
        if i == len(edges) - 2:          # last bin: include max
            b = int(data[-1])
        obs = int(np.sum((data >= a) & (data <= b)))
        p_a = nb_cdf(a - 1, r, p) if a > 0 else 0.0
        p_b = nb_cdf(b,     r, p)
        raw_obs.append(obs)
        raw_exp.append(n * max(0.0, p_b - p_a))

    # Forward-merge bins with expected < 5
    obs_m, exp_m = [], []
    acc_o, acc_e = 0.0, 0.0
    for o, e in zip(raw_obs, raw_exp):
        acc_o += o;  acc_e += e
        if acc_e >= 5.0:
            obs_m.append(acc_o);  exp_m.append(acc_e)
            acc_o, acc_e = 0.0, 0.0
    if acc_e > 0:
        if obs_m:
            obs_m[-1] += acc_o;  exp_m[-1] += acc_e
        else:
            obs_m.append(acc_o);  exp_m.append(acc_e)

    obs_m = np.array(obs_m);  exp_m = np.array(exp_m)
    # Renormalise expected to match observed total (handles tail probability)
    exp_m = exp_m * (obs_m.sum() / exp_m.sum())
    df    = max(1, len(obs_m) - 1 - ddof)
    chi2_s, _ = _cs(obs_m, f_exp=exp_m)
    from scipy.stats import chi2 as _chi2_dist
    pval  = float(_chi2_dist.sf(chi2_s, df))
    return float(chi2_s), pval, df, int(len(obs_m))

def ks_stat(data, r, p):
    """KS statistic D = sup_x |F_n(x) - F_NB(x)|."""
    data = np.sort(np.asarray(data, dtype=float))
    n    = len(data)
    xs   = np.unique(data).astype(int)
    D    = 0.0
    for x in xs:
        fn_lo = float(np.sum(data <  x)) / n   # F_n(x-1) for discrete
        fn_hi = float(np.sum(data <= x)) / n   # F_n(x)
        f_nb  = nb_cdf(x, r, p)
        D = max(D, abs(f_nb - fn_hi), abs(f_nb - fn_lo))
    return D


# ── Collect data ──────────────────────────────────────────────────────────────

print("Running 50 replicates …")
all_counts = {s: [] for s in SIZES}
for rep in range(REPS):
    seed = 99000 + rep    # identical seeds to pa_circuit_sizes.py
    sup, kv, rf = run_and_collect(N_STEPS, ALPHA, LAMBDA, GAMMA, BETA, START_R, seed)
    counts = count_by_size(sup, kv, rf, SIZES)
    for s in SIZES:
        all_counts[s].append(counts[s])
    if (rep + 1) % 10 == 0:
        print(f"  rep {rep+1:2d}/50  done")

count_arr = {s: np.array(all_counts[s]) for s in SIZES}


# ── Fit NB, bootstrap CIs, and GoF ───────────────────────────────────────────

rng_boot = np.random.default_rng(7777)
fits = {}

print("\nFitting NB and running bootstrap …")
for s in SIZES:
    arr          = count_arr[s]
    r_hat, p_hat, ok = fit_nb(arr)
    ci_lo, ci_hi = bootstrap_r_ci(arr, n_boot=N_BOOT, rng=rng_boot)
    chi2_s, chi2_p, df, nb_used = chi2_gof(arr, r_hat, p_hat)
    ks_d         = ks_stat(arr, r_hat, p_hat)
    mu_hat       = arr.mean()
    var_hat      = arr.var(ddof=1)
    # NB implied variance: mu + mu²/r
    var_nb       = mu_hat + mu_hat ** 2 / r_hat if r_hat > 0 else float("nan")
    fits[s] = dict(r=r_hat, p=p_hat, mu=mu_hat,
                   ci_lo=ci_lo, ci_hi=ci_hi,
                   chi2=chi2_s, chi2_p=chi2_p, df=df, nb_bins=nb_used,
                   ks=ks_d, var_emp=var_hat, var_nb=var_nb,
                   converged=ok)
    print(f"  s={s:2d}  r̂={r_hat:.3f} [{ci_lo:.3f},{ci_hi:.3f}]  "
          f"p̂={p_hat:.5f}  χ²({df})={chi2_s:.2f}  p={chi2_p:.4f}  "
          f"KS={ks_d:.4f}")


# ── Summary table ─────────────────────────────────────────────────────────────

print(f"\n{'s':>4}  {'r̂':>8}  {'95% CI':>18}  {'p̂':>9}  "
      f"{'μ̂':>10}  {'var_NB':>12}  {'var_emp':>12}  "
      f"{'χ²':>8}  {'df':>3}  {'p_χ²':>8}  {'KS D':>8}")
print("─" * 115)
for s in SIZES:
    f = fits[s]
    print(f"{s:>4}  {f['r']:>8.3f}  "
          f"[{f['ci_lo']:>6.3f}, {f['ci_hi']:>6.3f}]  "
          f"{f['p']:>9.6f}  {f['mu']:>10.1f}  "
          f"{f['var_nb']:>12.0f}  {f['var_emp']:>12.0f}  "
          f"{f['chi2']:>8.2f}  {f['df']:>3d}  "
          f"{f['chi2_p']:>8.4f}  {f['ks']:>8.4f}")


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

HIST_COLOR = "#2c7bb6"
NB_COLOR   = "#d7191c"
FILL_COLOR = "#fdae61"

for idx, s in enumerate(SIZES):
    ax  = axes[idx]
    arr = count_arr[s]
    f   = fits[s]
    r_hat, p_hat = f["r"], f["p"]
    mu   = f["mu"]

    # ── Histogram ────────────────────────────────────────────────────────
    n_bins = max(10, min(30, len(np.unique(arr))))
    cnts, bins, _ = ax.hist(arr, bins=n_bins, density=True,
                             color=HIST_COLOR, alpha=0.65, edgecolor="white",
                             label="Empirical (50 reps)")

    # ── NB density overlay ────────────────────────────────────────────────
    # Evaluate NB PMF on a fine grid over [min, max], smooth with bw
    x_lo = max(0, int(arr.min()))
    x_hi = int(arr.max())
    # For large ranges, sample 800 evenly spaced integers
    grid_size = min(x_hi - x_lo + 1, 1200)
    ks   = np.linspace(x_lo, x_hi, grid_size).astype(int)
    pmf  = np.exp(nb_logpmf(ks, r_hat, p_hat))
    # For plotting as continuous density: PMF(k) ≈ PDF evaluated at k
    ax.plot(ks, pmf, "-", color=NB_COLOR, lw=2.5,
            label=fr"NB($\hat r={r_hat:.2f},\ \hat p={p_hat:.4f}$)")
    ax.fill_between(ks, 0, pmf, color=FILL_COLOR, alpha=0.25)

    # ── Annotations ───────────────────────────────────────────────────────
    chi2_str = (f"χ²({f['df']})={f['chi2']:.1f}, p={f['chi2_p']:.3f}"
                if np.isfinite(f["chi2"]) else "χ²: n/a")
    ci_str   = f"95% CI: [{f['ci_lo']:.2f}, {f['ci_hi']:.2f}]"
    ax.set_title(
        fr"$s = {s}$" + "\n"
        fr"$\hat r = {r_hat:.3f}$,  $\hat\mu = {mu:.0f}$" + "\n"
        f"{chi2_str}     KS={f['ks']:.4f}\n{ci_str}",
        fontsize=9)
    ax.set_xlabel(fr"$N_{{{s}}}$  (circuit count)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

fig.suptitle(
    r"Negative Binomial fit to $N_s$: "
    r"$\gamma=0$, $\lambda=0.05$, $\alpha=0.3$, $\beta=0.5$, "
    r"$k_t=\lfloor 0.3\,r_t\rfloor$, $n_{\rm steps}=2000$, 50 reps"
    "\n"
    r"NB$(r,p)$: $\mu = r(1-p)/p$,  $\sigma^2 = \mu + \mu^2/r$  "
    r"(small $r$ = few effective independent hubs)",
    fontsize=11)
plt.tight_layout()
path = os.path.join(OUT, "circuit_nb_fit.png")
fig.savefig(path, dpi=150)
print(f"\n  → {path}")

# ── r̂ trend plot ─────────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

s_arr  = np.array(SIZES)
r_arr  = np.array([fits[s]["r"]    for s in SIZES])
cl_arr = np.array([fits[s]["ci_lo"] for s in SIZES])
ch_arr = np.array([fits[s]["ci_hi"] for s in SIZES])
ks_arr = np.array([fits[s]["ks"]   for s in SIZES])
p_arr  = np.array([fits[s]["chi2_p"] for s in SIZES])

ax = axes2[0]
ax.errorbar(s_arr, r_arr, yerr=[r_arr - cl_arr, ch_arr - r_arr],
            fmt="o-", color="#2c7bb6", lw=2, ms=9, capsize=6,
            label=r"$\hat r$ ± 95% boot CI")
ax.set_xlabel("Circuit size $s$", fontsize=12)
ax.set_ylabel(r"NB dispersion parameter $\hat r$", fontsize=12)
ax.set_title(r"$\hat r$ vs $s$: effective hub count", fontsize=12)
ax.set_xticks(SIZES)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Annotate interpretation
for i, s in enumerate(SIZES):
    ax.annotate(f"{r_arr[i]:.2f}", (s_arr[i], r_arr[i]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)

ax = axes2[1]
bar_colors = ["#2ca25f" if p > 0.05 else "#de2d26" for p in p_arr]
ax.bar(s_arr, p_arr, width=0.8, color=bar_colors, edgecolor="k", linewidth=0.8,
       alpha=0.8)
ax.axhline(0.05, color="k", ls="--", lw=1.5, label="α=0.05")
ax.set_xlabel("Circuit size $s$", fontsize=12)
ax.set_ylabel(r"$\chi^2$ p-value", fontsize=12)
ax.set_title("Chi-squared GoF p-value\n(green = NB not rejected at 5%)", fontsize=12)
ax.set_xticks(SIZES)
ax.set_ylim(0, max(1.0, p_arr.max() * 1.2))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

fig2.suptitle(
    r"NB fit quality: dispersion $\hat r$ and GoF across circuit sizes",
    fontsize=12)
plt.tight_layout()
path2 = os.path.join(OUT, "circuit_nb_fit_summary.png")
fig2.savefig(path2, dpi=150)
print(f"  → {path2}")
print("""
NB interpretation:
  r̂ = effective number of independent PA hubs driving N_s
  Small r̂ → few dominant hubs → high burstiness / overdispersion
  var_NB = μ + μ²/r  should match var_emp if NB is correct
  χ² p > 0.05 → NB not rejected;  KS D small → good CDF fit
""")
