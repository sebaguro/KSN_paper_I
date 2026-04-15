"""
KSN Figure Generator — Option A (Direct Primary Energy Method)
===============================================================
Reads energy data from owid-energy-data.json (direct method)
so that figures and notebook use IDENTICAL data.

Generates two versions of the KSN analysis figure:
  1. KSN_titled.png   — with panel titles (for checking)
  2. KSN_notitle.png  — no titles (clean for LaTeX import)

Requirements:
    pip install numpy matplotlib scipy astropy

Run from the directory containing owid-energy-data.json:
    python3 ksn_figure_optA.py

Author: S. Gurovich / Claude (Anthropic)
Data sources:
    Energy:   OWID / EI Statistical Review of World Energy 2025
              (direct primary energy accounting method)
    Hashrate: Blockchain.com / NASDAQ BCHAIN/HRATE
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy import stats
import json
import os

# ── Try astropy for MJD; fall back to formula if not available ────────────────
try:
    from astropy.time import Time
    def yr2mjd(y):
        y = np.atleast_1d(np.array(y, dtype=float))
        out = np.zeros_like(y)
        for i, yi in enumerate(y):
            out[i] = Time(f"{int(yi)}-07-01", format='iso', scale='utc').mjd
        return out if len(out) > 1 else float(out[0])
    print("Using astropy for MJD conversion")
except ImportError:
    def yr2mjd(y):
        return (np.array(y, dtype=float) - 1858.0) * 365.25 - 45 + 182
    print("astropy not found — using MJD formula approximation")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
YEAR_ZERO    = 1964
# Eq. (1) of the paper: P = E * 1e12 / (365.25 * 24)  [Julian year]
TWH_TO_WATTS = 1e12 / (365.25 * 24)
KARD_R       = 0.01                    # Kardashev 1% growth rate
KARD_TYPE1   = 4.0e12                  # Kardashev 1964 Type I  (W)
KARD_TYPE2   = 4.0e26                  # Kardashev 1964 Type II (W)
SAGAN_TYPE2  = 1.0e26                  # Sagan 1973 Type II     (W)
INSOLATION   = 1.74e17                 # Solar insolation at Earth (W)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD ENERGY DATA FROM OWID JSON
# ═══════════════════════════════════════════════════════════════════════════════
json_path = 'owid-energy-data.json'
if not os.path.exists(json_path):
    print(f"ERROR: {json_path} not found in current directory.")
    print("Please run this script from the directory containing the OWID JSON.")
    exit(1)

with open(json_path) as f:
    owid = json.load(f)

world_data = owid['World']['data']

energy_data = {}
for i in range(65, 125):
    entry = world_data[i]
    energy_data[entry['year']] = np.float64(entry['primary_energy_consumption'])

print(f"Loaded {len(energy_data)} energy data points from {json_path}")
print(f"  Method: direct primary energy (current OWID default)")
print(f"  E({min(energy_data)}) = {energy_data[min(energy_data)]:.2f} TWh")
print(f"  E({max(energy_data)}) = {energy_data[max(energy_data)]:.2f} TWh")

# ═══════════════════════════════════════════════════════════════════════════════
# 2b. BITCOIN HASHRATE DATA (no OWID equivalent — hardcoded)
# ═══════════════════════════════════════════════════════════════════════════════
# 2009–2013: Annual averages derived from on-chain difficulty via
#            H = D * 2^32 / 600, using the logarithmic mean of the
#            Jan 1 and Dec 31 difficulty values (appropriate for
#            exponentially growing quantities).
# 2014–2024: Yearly averages from Blockchain.com / NASDAQ BCHAIN/HRATE.
hashrate_data = {
    2009: 7.16e6,     # CPU era — difficulty = 1 all year
    2010: 1.08e10,    # GPU era begins ~mid 2010
    2011: 1.87e12,    # GPU era
    2012: 1.50e13,    # Late GPU era
    2013: 1.44e15,    # ASIC era begins Jan 2013
    2014: 2.0e17,     # ASIC era
    2015: 4.5e17,
    2016: 1.6e18,
    2017: 6.0e18,
    2018: 3.5e19,
    2019: 8.0e19,
    2020: 1.3e20,
    2021: 1.5e20,     # China mining ban mid-2021 caused dip
    2022: 2.3e20,
    2023: 4.0e20,
    2024: 6.2e20,
}

# ═══════════════════════════════════════════════════════════════════════════════
# 3. BUILD ARRAYS
# ═══════════════════════════════════════════════════════════════════════════════
e_years = np.array(sorted(energy_data.keys()))
e_watts = np.array([energy_data[y] for y in e_years]) * TWH_TO_WATTS
t_e     = (e_years - YEAR_ZERO).astype(float)

common   = sorted(set(energy_data) & set(hashrate_data))
jh_years = np.array(common)
jh_e     = np.array([energy_data[y] * TWH_TO_WATTS for y in common])
jh_h     = np.array([hashrate_data[y] for y in common])
jh       = jh_e / jh_h   # KarNak units: J/Hash

print(f"\nEnergy data:   {e_years[0]}–{e_years[-1]}  ({len(e_years)} points)")
print(f"P({e_years[0]})  = {e_watts[0]:.4e} W = {e_watts[0]/1e12:.2f} TW")
print(f"P({e_years[-1]}) = {e_watts[-1]:.4e} W = {e_watts[-1]/1e12:.2f} TW")
print(f"Hashrate data: {jh_years[0]}–{jh_years[-1]}  ({len(jh_years)} points)")
print(f"J/Hash range:  {jh[0]:.3e} – {jh[-1]:.3e}  ({len(jh)} points, "
      f"{np.log10(jh[0]/jh[-1]):.1f} orders of magnitude)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL FITS
# ═══════════════════════════════════════════════════════════════════════════════
def lin(t, a, b):    return a + b * t
def expm(t, a0, r):  return a0 * np.exp(r * t)

p_lin, _ = curve_fit(lin,  t_e, e_watts,
                     p0=[e_watts[0], (e_watts[-1]-e_watts[0])/t_e[-1]])
p_exp, _ = curve_fit(expm, t_e, e_watts,
                     p0=[e_watts[0], 0.02], maxfev=20000)
a_lin, b_lin = p_lin
a0_exp, r_exp = p_exp

def r2(obs, pred):
    return 1 - np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2)

r2_lin = r2(e_watts, lin(t_e, *p_lin))
r2_exp = r2(e_watts, expm(t_e, *p_exp))

print(f"\nLinear fit:      a={a_lin:.3e} W,  b={b_lin:.3e} W/yr,  R²={r2_lin:.6f}")
print(f"Exponential fit: a0={a0_exp:.3e} W,  r={r_exp*100:.3f}%/yr,  R²={r2_exp:.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════
diffs                      = np.diff(e_watts)
resid                      = e_watts - lin(t_e, *p_lin)
sw_stat, sw_p              = stats.shapiro(resid)
sw_diffs_stats, sw_diffs_p = stats.shapiro(diffs)
skew_diffs                 = stats.skew(diffs)

t2_lin  = (KARD_TYPE2 - a_lin) / b_lin
t2_exp  = np.log(KARD_TYPE2 / a0_exp) / r_exp
t2_kard = np.log(KARD_TYPE2 / KARD_TYPE1) / KARD_R
univ    = 13.8e9
ratio   = t2_lin / univ

print(f"\nShapiro-Wilk on ΔP:     W={sw_diffs_stats:.4f},  p={sw_diffs_p:.6f},  skewness={skew_diffs:.4f}")
print(f"Shapiro-Wilk on resid:  W={sw_stat:.4f},  p={sw_p:.6f}")
print(f"Type II timescale — Linear:      {t2_lin:.3e} yrs = {ratio:.2e} × H₀⁻¹")
print(f"Type II timescale — Exp r-free:  year {YEAR_ZERO+t2_exp:.0f} CE")
print(f"Type II timescale — Kardashev:   year {YEAR_ZERO+t2_kard:.0f} CE")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PLOT FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
BLUE = "#2166ac"; RED = "#d6604d"; ORG = "#f4a582"
GRN  = "#1a9641"; BLK = "#111111"

CPU_START  = 2009.0
GPU_START  = 2010.5
ASIC_START = 2013.0


def make_figure(with_titles=True):
    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

    # ── Extended time axis for extrapolation ──────────────────────────────────
    XMIN, XMAX = 1964, 8000
    t_ext  = np.linspace(0, XMAX - YEAR_ZERO, 500000)
    yr_ext = t_ext + YEAR_ZERO

    # ── PANEL A — Energy + models + civilisation limits ───────────────────────
    ax1 = fig.add_subplot(gs[0, :])

    ax1.scatter(e_years, e_watts, color=BLK, s=35, zorder=6,
                label='OWID / EI World data  (1965–2024)')
    ax1.plot(yr_ext, lin(t_ext, *p_lin), '--', color=BLUE, lw=2.5,
             label=f'Model 1: Linear  ($R^2={r2_lin:.4f}$)')
    ax1.plot(yr_ext, expm(t_ext, *p_exp), '-', color=RED, lw=2.5,
             label=f'Model 2: Exp  $r={r_exp*100:.2f}\\%$/yr  ($R^2={r2_exp:.4f}$)')
    ax1.plot(yr_ext, expm(t_ext, a0_exp, KARD_R), ':', color=ORG, lw=2.2,
             label='Kardashev $r=1\\%$/yr')

    ax1.axhline(KARD_TYPE2,  color='darkviolet',   lw=2.0, ls='-.',
                label='Type II  Kardashev 1964  ($4\\times10^{26}$ W)')
    ax1.axhline(SAGAN_TYPE2, color='mediumorchid', lw=1.8, ls=(0,(4,1,1,1)),
                label='Type II  Sagan 1973  ($10^{26}$ W)')
    ax1.axhline(INSOLATION,  color='darkorange',   lw=1.8, ls='--',
                label='Solar insolation at Earth  ($1.74\\times10^{17}$ W)')
    ax1.axhline(KARD_TYPE1,  color='olive',        lw=1.4, ls=':',
                label='Type I  Kardashev 1964  ($4\\times10^{12}$ W)')

    ax1.set_yscale('log')
    ax1.set_ylim(1e9, 1e28)
    ax1.set_xlim(XMIN, XMAX)
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Global Energy Production  (W)', fontsize=11)
    ax1.legend(fontsize=10, loc='lower center', ncol=3,
               framealpha=0.92, handlelength=3)
    ax1.grid(True, alpha=0.05, which='both')

    # Top axis: MJD
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(XMIN, XMAX)
    year_ticks = np.array([2000, 2500, 3000, 3500, 4000,
                           4500, 5000, 5500, 6000, 6500, 7000, 7500])
    mjd_labels = [f'{int(yr2mjd(y))}' for y in year_ticks]
    ax1_top.set_xticks(year_ticks)
    ax1_top.set_xticklabels(mjd_labels, fontsize=9, rotation=30, ha='left')
    ax1_top.set_xlabel('Modified Julian Date', fontsize=10)

    if with_titles:
        ax1.set_title('Global Energy Production as a function of time.',
                      fontweight='bold')

    # ── PANEL B — KSN J/Hash data + mining era lines ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])

    ax3.scatter(jh_years, jh, color=GRN, s=90, zorder=6,
                marker='+', linewidths=2.5,
                label=r'$\mathscr{B}(t) \;=\; \frac{P(t)}{\mathcal{H}(t)}$  (KarNak)')

    ax3.axvline(CPU_START,  color='forestgreen', lw=2.0, ls='--', alpha=0.85,
                label='CPU era begins (Jan 2009)')
    ax3.axvline(GPU_START,  color='royalblue',   lw=2.0, ls='--', alpha=0.85,
                label='GPU era begins ($\\sim$mid 2010)')
    ax3.axvline(ASIC_START, color='crimson',     lw=2.0, ls='--', alpha=0.85,
                label='ASIC era begins (Jan 2013)')

    ax3.axvspan(CPU_START,  GPU_START,          alpha=0.07, color='forestgreen')
    ax3.axvspan(GPU_START,  ASIC_START,         alpha=0.07, color='royalblue')
    ax3.axvspan(ASIC_START, jh_years[-1]+0.5,   alpha=0.07, color='crimson')

    ax3.set_yscale('log')
    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_ylabel(r'$\mathscr{B}(t) \;=\; \frac{P(t)}{\mathcal{H}(t)}$ (KarNak)',
                   fontsize=11)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.2, which='both')

    if with_titles:
        ax3.set_title('Panel B -- Kardashev--Sagan--Nakamoto Model\n'
                      r'$\mathscr{B}(t) \;=\; \frac{P(t)}{\mathcal{H}(t)}$  in KarNak units',
                      fontweight='bold', color=GRN)

    # ── PANEL C — Year-on-year differences ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    diffs_plot = np.diff(e_watts)
    diff_years = e_years[1:]

    ax4.scatter(diff_years, diffs_plot/1e9, color=BLUE, s=28, zorder=5)
    ax4.axhline(0, color='k', lw=1)

    events = [
        (1980, '1980: Oil crisis',   -8,   -60),
        (1981, '1981: Recession',     5,   -80),
        (1982, '1982: Recession',    -8,  -100),
        (2009, '2009: GFC',           5,   120),
        (2020, '2020: COVID-19',    -12,  -120),
    ]
    for yr_h, label_h, xoff, yoff in events:
        idx  = list(diff_years).index(yr_h)
        y_pt = diffs_plot[idx] / 1e9
        ax4.annotate(
            label_h,
            xy=(yr_h, y_pt),
            xytext=(yr_h + xoff, y_pt + yoff),
            fontsize=8.5, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.4),
            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                      ec='red', alpha=0.9)
        )

    ax4.set_xlabel('Year', fontsize=11)
    ax4.set_ylabel(r'$\Delta P_i$  ($\times10^9$ W)', fontsize=11)
    ax4.grid(True, alpha=0.2)

    ax4.text(0.05, 0.03,
             f'$\\Delta P_i$  Shapiro--Wilk:\n'
             f'$W={sw_diffs_stats:.3f}$,  $p={sw_diffs_p:.4f}$\n'
             f'Skewness $= {skew_diffs:.3f}$\n'
             'Rejects normality ($\\alpha=0.05$)',
             transform=ax4.transAxes, va='bottom', fontsize=9,
             bbox=dict(boxstyle='round', fc='white', alpha=0.85))

    if with_titles:
        ax4.set_title('Panel C -- Year-on-year differences: $\\Delta P_i$',
                      fontweight='bold')
        plt.suptitle(
            'The Kardashev Conundrum and the KSN Renormalization\n'
            'OWID / EI Statistical Review 2025 (direct method)  |  '
            'Hashrate: Blockchain.com / NASDAQ BCHAIN/HRATE',
            fontsize=11, fontweight='bold', y=1.01)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SAVE BOTH VERSIONS
# ═══════════════════════════════════════════════════════════════════════════════
outdir = os.path.dirname(os.path.abspath(__file__))

fig1 = make_figure(with_titles=True)
p1   = os.path.join(outdir, 'KSN_titled.png')
fig1.savefig(p1, dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f"\nSaved: {p1}")

fig2 = make_figure(with_titles=False)
p2   = os.path.join(outdir, 'KSN_notitle.png')
fig2.savefig(p2, dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f"Saved: {p2}")

print("\nDone. Upload KSN_notitle.png to Overleaf for LaTeX import.")
