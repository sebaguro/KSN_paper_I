#!/usr/bin/env python3
"""
Independent MCMC verification for KSN Paper I.
Reads directly from owid-energy-data.json (direct method).
Tests both year-based and MJD-based fits to confirm invariance.

Run on Sersic:  python ksn_mcmc_verify.py
"""
import json
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import shapiro, skew

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA — same logic as ksn_figure_optA.py
# ═══════════════════════════════════════════════════════════════════════
with open('owid-energy-data.json', 'r') as f:
    owid = json.load(f)

world_data = owid['World']['data']

energy_data = {}
for entry in world_data:
    yr = entry.get('year')
    val = entry.get('primary_energy_consumption')
    if yr and val and 1965 <= yr <= 2024:
        energy_data[yr] = val

years = np.array(sorted(energy_data.keys()), dtype=float)
E_twh = np.array([energy_data[int(y)] for y in years])
N = len(years)

# Convert to watts (direct method)
TWH_TO_W = 1e12 / (365.25 * 24)
P = E_twh * TWH_TO_W

# Time in years since 1964
t_yr = years - 1964.0

# Time in MJD (using manual conversion, no astropy needed)
# MJD of Jan 1 of each year: MJD = 51544.5 + (year - 2000)*365.25
# More precisely: use Julian Day formula
def year_to_mjd(yr):
    """MJD of Jan 1 of given year (approximate, good to ~0.1 day)"""
    # Julian Day Number for Jan 1
    a = (14 - 1) // 12
    y = int(yr) + 4800 - a
    m = 1 + 12*a - 3
    jdn = 1 + (153*m + 2)//5 + 365*y + y//4 - y//100 + y//400 - 32045
    return jdn - 2400000.5

mjd = np.array([year_to_mjd(y) for y in years])
t_mjd = mjd - mjd[0]  # offset from first point

print("=" * 70)
print("KSN Paper I — Independent MCMC Verification")
print("=" * 70)
print(f"\nData: {N} points, {int(years[0])}–{int(years[-1])}")
print(f"E(1965) = {E_twh[0]:.2f} TWh")
print(f"E(2024) = {E_twh[-1]:.2f} TWh")
print(f"P(1965) = {P[0]:.4e} W = {P[0]/1e12:.2f} TW")
print(f"P(2024) = {P[-1]:.4e} W = {P[-1]/1e12:.2f} TW")

# ═══════════════════════════════════════════════════════════════════════
# 2. OLS LINEAR FIT (in years)
# ═══════════════════════════════════════════════════════════════════════
coeffs = np.polyfit(t_yr, P, 1)
b_lin, a_lin = coeffs
P_lin = a_lin + b_lin * t_yr
SS_res_lin = np.sum((P - P_lin)**2)
SS_tot = np.sum((P - np.mean(P))**2)
R2_lin = 1 - SS_res_lin / SS_tot

print(f"\n{'─'*70}")
print("LINEAR FIT (years)")
print(f"  a = {a_lin:.4e} W")
print(f"  b = {b_lin:.4e} W/yr")
print(f"  R² = {R2_lin:.6f}")

# ═══════════════════════════════════════════════════════════════════════
# 3. EXPONENTIAL FIT via curve_fit (years) — for comparison
# ═══════════════════════════════════════════════════════════════════════
def exp_model(t, a0, r):
    return a0 * np.exp(r * t)

popt, pcov = curve_fit(exp_model, t_yr, P, p0=[P[0], 0.02], maxfev=50000)
a0_cf, r_cf = popt
P_exp_cf = exp_model(t_yr, *popt)
R2_exp_cf = 1 - np.sum((P - P_exp_cf)**2) / SS_tot

print(f"\n{'─'*70}")
print("EXPONENTIAL FIT via curve_fit (years)")
print(f"  a0 = {a0_cf:.4e} W")
print(f"  r  = {r_cf*100:.4f} %/yr")
print(f"  R² = {R2_exp_cf:.6f}")

# ═══════════════════════════════════════════════════════════════════════
# 4. MCMC — Metropolis-Hastings (years)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("MCMC INFERENCE (years)")

np.random.seed(42)

def log_likelihood(params, t, P_obs):
    a0, r, log_sigma = params
    sigma = np.exp(log_sigma)
    model = a0 * np.exp(r * t)
    resid = P_obs - model
    return -0.5 * N * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(resid**2) / sigma**2

def log_prior(params):
    a0, r, log_sigma = params
    # Prior on r: N(0.01, 0.02^2) — centred on Kardashev
    lp = -0.5 * ((r - 0.01) / 0.02)**2
    # Broad priors on a0 and sigma
    if a0 < 0 or a0 > 1e14:
        return -np.inf
    if log_sigma < 20 or log_sigma > 35:
        return -np.inf
    return lp

def log_posterior(params, t, P_obs):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, t, P_obs)

# Initial guess from curve_fit
sigma_init = np.std(P - exp_model(t_yr, *popt))
theta = np.array([a0_cf, r_cf, np.log(sigma_init)])

n_steps = 75000
burn_in = 15000
chain = np.zeros((n_steps, 3))
chain[0] = theta

# Proposal widths
proposal_sigma = np.array([1e10, 0.0002, 0.05])
n_accept = 0

for i in range(1, n_steps):
    proposal = chain[i-1] + np.random.normal(0, proposal_sigma)
    lp_new = log_posterior(proposal, t_yr, P)
    lp_old = log_posterior(chain[i-1], t_yr, P)
    
    if np.log(np.random.uniform()) < lp_new - lp_old:
        chain[i] = proposal
        n_accept += 1
    else:
        chain[i] = chain[i-1]

accept_rate = n_accept / n_steps
samples = chain[burn_in:]
r_samples = samples[:, 1] * 100  # convert to percent

r_mean = np.mean(r_samples)
r_std = np.std(r_samples)
r_lo = np.percentile(r_samples, 2.5)
r_hi = np.percentile(r_samples, 97.5)

print(f"  Acceptance rate: {accept_rate:.3f}")
print(f"  Posterior r: {r_mean:.2f} ± {r_std:.2f} %/yr")
print(f"  95% CI: [{r_lo:.2f}%, {r_hi:.2f}%]")
print(f"  Kardashev 1% is {(r_mean - 1.0)/r_std:.1f}σ from posterior mean")

# ═══════════════════════════════════════════════════════════════════════
# 5. REPEAT IN MJD to confirm invariance
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("EXPONENTIAL FIT via curve_fit (MJD)")

popt_mjd, _ = curve_fit(exp_model, t_mjd, P, p0=[P[0], 0.02/365.25], maxfev=50000)
a0_mjd, r_mjd = popt_mjd
P_exp_mjd = exp_model(t_mjd, *popt_mjd)
R2_exp_mjd = 1 - np.sum((P - P_exp_mjd)**2) / SS_tot

# Convert r from per-day to per-year
r_mjd_yr = r_mjd * 365.25

print(f"  a0 = {a0_mjd:.4e} W")
print(f"  r  = {r_mjd*100:.6f} %/day = {r_mjd_yr*100:.4f} %/yr")
print(f"  R² = {R2_exp_mjd:.6f}")
print(f"  R² difference (yr vs MJD): {abs(R2_exp_cf - R2_exp_mjd):.2e}")

# Linear fit in MJD
coeffs_mjd = np.polyfit(t_mjd, P, 1)
b_mjd, a_mjd = coeffs_mjd
P_lin_mjd = a_mjd + b_mjd * t_mjd
R2_lin_mjd = 1 - np.sum((P - P_lin_mjd)**2) / SS_tot
b_mjd_yr = b_mjd * 365.25

print(f"\n{'─'*70}")
print("LINEAR FIT (MJD)")
print(f"  a = {a_mjd:.4e} W")
print(f"  b = {b_mjd:.4e} W/day = {b_mjd_yr:.4e} W/yr")
print(f"  R² = {R2_lin_mjd:.6f}")
print(f"  R² difference (yr vs MJD): {abs(R2_lin - R2_lin_mjd):.2e}")

# ═══════════════════════════════════════════════════════════════════════
# 6. SHAPIRO-WILK and SKEWNESS
# ═══════════════════════════════════════════════════════════════════════
dP = np.diff(P)
sw_stat, sw_p = shapiro(dP)
sk = skew(dP)

print(f"\n{'─'*70}")
print("SHAPIRO-WILK on ΔP")
print(f"  W = {sw_stat:.4f}")
print(f"  p = {sw_p:.6f}")
print(f"  Skewness = {sk:.4f}")
print(f"  Normality rejected at α=0.05: {sw_p < 0.05}")

# ═══════════════════════════════════════════════════════════════════════
# 7. TYPE II TIMESCALES
# ═══════════════════════════════════════════════════════════════════════
L_sun = 3.828e26
t_typeII_lin = (L_sun - a_lin) / b_lin
H0_inv = 13.8e9

a0_mcmc = np.mean(samples[:, 0])
r_mcmc = np.mean(samples[:, 1])
t_typeII_exp = np.log(L_sun / a0_mcmc) / r_mcmc
year_typeII_exp = 1964 + t_typeII_exp

t_typeII_kard = np.log(L_sun / P[0]) / 0.01
year_typeII_kard = 1964 + t_typeII_kard

print(f"\n{'─'*70}")
print("TYPE II TIMESCALES")
print(f"  Linear:    {t_typeII_lin:.3e} yr = {t_typeII_lin/H0_inv:.2e} × H₀⁻¹")
print(f"  Exp (MCMC): year {year_typeII_exp:.0f} CE")
print(f"  Kardashev:  year {year_typeII_kard:.0f} CE")

# ═══════════════════════════════════════════════════════════════════════
# 8. WAIC COMPARISON — Full posterior-predictive (Watanabe 2010)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("WAIC COMPARISON (full posterior-predictive)")

# For the EXPONENTIAL model, we already have MCMC samples.
# WAIC = -2 * (lppd - p_waic)
# where lppd = Σᵢ log( 1/S Σₛ p(yᵢ|θₛ) )       [log pointwise predictive density]
#       p_waic = Σᵢ Var_s[ log p(yᵢ|θₛ) ]         [effective number of parameters]

# Step 1: Compute pointwise log-likelihoods for each posterior sample
#         for the exponential model
n_samples = len(samples)
# Thin to max 5000 samples for speed
thin = max(1, n_samples // 5000)
samples_thin = samples[::thin]
S = len(samples_thin)

# Exponential model: log p(yᵢ | a0_s, r_s, sigma_s)
# Each sample has (a0, r, log_sigma)
ll_exp_matrix = np.zeros((S, N))  # S samples x N data points
for s in range(S):
    a0_s, r_s, log_sigma_s = samples_thin[s]
    sigma_s = np.exp(log_sigma_s)
    model_s = a0_s * np.exp(r_s * t_yr)
    ll_exp_matrix[s] = -0.5 * np.log(2*np.pi*sigma_s**2) - 0.5*((P - model_s)/sigma_s)**2

# lppd_exp = Σᵢ log( mean_s( exp(ll_is) ) )
# Use log-sum-exp trick for numerical stability
max_ll = np.max(ll_exp_matrix, axis=0)
lppd_exp = np.sum(max_ll + np.log(np.mean(np.exp(ll_exp_matrix - max_ll), axis=0)))

# p_waic_exp = Σᵢ Var_s[ ll_is ]
p_waic_exp = np.sum(np.var(ll_exp_matrix, axis=0, ddof=1))

WAIC_exp = -2 * (lppd_exp - p_waic_exp)

# Step 2: For the LINEAR model, we need an MCMC chain too.
# Run a quick MCMC for the linear model
print("  Running linear MCMC...")

def log_likelihood_lin(params, t, P_obs):
    a, b, log_sigma = params
    sigma = np.exp(log_sigma)
    model = a + b * t
    resid = P_obs - model
    return -0.5 * N * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(resid**2) / sigma**2

def log_prior_lin(params):
    a, b, log_sigma = params
    if a < 0 or a > 1e14:
        return -np.inf
    if b < 0 or b > 1e13:
        return -np.inf
    if log_sigma < 20 or log_sigma > 35:
        return -np.inf
    return 0.0  # flat prior

def log_posterior_lin(params, t, P_obs):
    lp = log_prior_lin(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_lin(params, t, P_obs)

sigma_lin_init = np.std(P - P_lin)
theta_lin = np.array([a_lin, b_lin, np.log(sigma_lin_init)])

n_steps_lin = 75000
burn_in_lin = 15000
chain_lin = np.zeros((n_steps_lin, 3))
chain_lin[0] = theta_lin

proposal_sigma_lin = np.array([5e9, 5e8, 0.05])
n_accept_lin = 0

for i in range(1, n_steps_lin):
    proposal = chain_lin[i-1] + np.random.normal(0, proposal_sigma_lin)
    lp_new = log_posterior_lin(proposal, t_yr, P)
    lp_old = log_posterior_lin(chain_lin[i-1], t_yr, P)
    
    if np.log(np.random.uniform()) < lp_new - lp_old:
        chain_lin[i] = proposal
        n_accept_lin += 1
    else:
        chain_lin[i] = chain_lin[i-1]

samples_lin = chain_lin[burn_in_lin:]
samples_lin_thin = samples_lin[::thin]
S_lin = len(samples_lin_thin)

print(f"  Linear MCMC acceptance rate: {n_accept_lin/n_steps_lin:.3f}")

# Pointwise log-likelihoods for linear model
ll_lin_matrix = np.zeros((S_lin, N))
for s in range(S_lin):
    a_s, b_s, log_sigma_s = samples_lin_thin[s]
    sigma_s = np.exp(log_sigma_s)
    model_s = a_s + b_s * t_yr
    ll_lin_matrix[s] = -0.5 * np.log(2*np.pi*sigma_s**2) - 0.5*((P - model_s)/sigma_s)**2

max_ll_lin = np.max(ll_lin_matrix, axis=0)
lppd_lin = np.sum(max_ll_lin + np.log(np.mean(np.exp(ll_lin_matrix - max_ll_lin), axis=0)))
p_waic_lin = np.sum(np.var(ll_lin_matrix, axis=0, ddof=1))

WAIC_lin = -2 * (lppd_lin - p_waic_lin)

dWAIC = WAIC_exp - WAIC_lin

print(f"  WAIC linear: {WAIC_lin:.1f}  (lppd={lppd_lin:.1f}, p_waic={p_waic_lin:.2f})")
print(f"  WAIC exp:    {WAIC_exp:.1f}  (lppd={lppd_exp:.1f}, p_waic={p_waic_exp:.2f})")
print(f"  ΔWAIC (exp - lin): {dWAIC:.1f}  (positive = linear preferred)")
print(f"  p_waic difference: {p_waic_exp - p_waic_lin:.2f} (exp has more effective params)")

# ═══════════════════════════════════════════════════════════════════════
# 9. SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print("SUMMARY — COMPARE WITH MANUSCRIPT VALUES")
print(f"{'═'*70}")
print(f"  P(1965) = {P[0]/1e12:.2f} TW              [paper: 4.95 TW]")
print(f"  P(2024) = {P[-1]/1e12:.2f} TW             [paper: 20.16 TW]")
print(f"  b = {b_lin:.3e} W/yr           [paper: 2.44e11]")
print(f"  R² (linear) = {R2_lin:.6f}          [paper: 0.987]")
print(f"  R² (exp) = {R2_exp_cf:.6f}             [paper: 0.987]")
print(f"  r (MCMC) = {r_mean:.2f} ± {r_std:.2f} %/yr   [paper: 2.01 ± 0.03]")
print(f"  95% CI = [{r_lo:.2f}%, {r_hi:.2f}%]        [paper: [1.94%, 2.08%]]")
print(f"  SW W = {sw_stat:.4f}                    [paper: 0.925]")
print(f"  SW p = {sw_p:.6f}                [paper: 0.0014]")
print(f"  Skewness = {sk:.4f}                [paper: -0.664]")
print(f"  ΔWAIC = {dWAIC:.1f}                     [paper: 5.5]")
print(f"  Type II (linear) = {t_typeII_lin:.3e} yr  [paper: ~1.6e15]")
print(f"  Type II (exp) = {year_typeII_exp:.0f} CE         [paper: ~3547]")
print(f"  Type II (Kard) = {year_typeII_kard:.0f} CE        [paper: ~5188]")
print(f"\n  x-axis invariance:")
print(f"  R²(yr) - R²(MJD) linear: {abs(R2_lin - R2_lin_mjd):.2e}")
print(f"  R²(yr) - R²(MJD) exp:    {abs(R2_exp_cf - R2_exp_mjd):.2e}")
print(f"  r(yr) vs r(MJD→yr):      {r_cf*100:.4f}% vs {r_mjd_yr*100:.4f}%")
