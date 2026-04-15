# Kardashev's Conundrum: KSN Paper I

**Statistical Falsification of the Standard Kardashev Model and the Kardashev–Sagan–Nakamoto Resolution**

S. Gurovich (IATE–OAC–UNC–CONICET / Western Sydney University)

Submitted to *Astronomy and Computing* (Elsevier), 2026.

---

## Summary

This repository contains the data, code, and manuscript source for KSN Paper I, which:

1. **Falsifies** the standard Kardashev 1% exponential energy growth model using 60 years of global energy production data (1965–2024) from Our World in Data.
2. **Identifies Kardashev's Conundrum**: no functional form fitted to energy production alone can simultaneously satisfy statistical adequacy and physical coherence.
3. **Proposes the KSN resolution**: renormalising power by the Bitcoin network hashrate to define the KSN state variable B(t) = P(t)/H(t) in units of J Hash⁻¹ (the KarNak unit).

## Key Results

| Quantity | Value |
|---|---|
| Posterior growth rate (MCMC) | r = 2.01 ± 0.03 % yr⁻¹ |
| 95% credible interval | [1.94%, 2.08%] |
| Linear slope | b = 2.44 × 10¹¹ W yr⁻¹ |
| R² (linear) | 0.987 |
| ΔWAIC (exp − linear) | 5.5 (linear preferred) |
| Shapiro–Wilk on ΔP | W = 0.925, p = 0.0014 |
| Type II timescale (linear) | ~1.6 × 10¹⁵ yr ≈ 1.2 × 10⁵ H₀⁻¹ |
| B(t) range | 2.15 × 10⁶ to 3.25 × 10⁻⁸ J/Hash (14 orders of magnitude) |

## Repository Structure

```
.
├── astroblock.tex            # Manuscript source (elsarticle format)
├── astroblock.bib             # BibTeX references
├── figs/
│   ├── KSN_panel1.png        # Figure 1: Energy production + models
│   ├── KSN_panel2.png        # Figure 3: KSN variable B(t)
│   └── KSN_panel3.png        # Figure 2: Year-over-year ΔP
├── ksn_figure_optA.py        # Figure generation script (reads OWID JSON)
├── ksn_mcmc_verify.py        # Independent MCMC verification script
└── README.md
```

## Data Sources

- **Energy data**: [Our World in Data](https://ourworldindata.org/energy) — `owid-energy-data.json`, direct primary energy method (EI Statistical Review 2025).
- **Bitcoin hashrate (2009–2013)**: Derived from on-chain difficulty via H = D × 2³²/600, using the logarithmic mean of Jan 1 and Dec 31 difficulty values.
- **Bitcoin hashrate (2014–2024)**: Blockchain.com / NASDAQ BCHAIN/HRATE annual averages.

## Reproducing the Results

1. Download `owid-energy-data.json` from [OWID GitHub](https://github.com/owid/energy-data).
2. Run the figure script:
   ```bash
   python ksn_figure_optA.py
   ```
   Requires: `numpy`, `scipy`, `matplotlib`, `astropy`, `json`

3. Run the independent MCMC verification:
   ```bash
   python ksn_mcmc_verify.py
   ```
   Requires: `numpy`, `scipy`

## Citation

If you use this work, please cite:

```bibtex
@article{Gurovich2026,
  author  = {Gurovich, S.},
  title   = {Kardashev's Conundrum: Statistical Falsification of the
             Standard Kardashev Model and the Kardashev--Sagan--Nakamoto
             Resolution},
  journal = {Astronomy and Computing},
  year    = {2026},
  note    = {Submitted}
}
```

## License

The manuscript and code are provided for review and reproducibility purposes.

## Contact

Sebastian Gurovich — sgurovich@unc.edu.ar