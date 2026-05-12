# Monte Carlo Physics Simulator

Three small Monte Carlo studies in one script, each a self-contained
demonstration of a different sampling technique applied to a physics problem.

## What it does

1. **Sampling `sin(θ)/2` on `[0, π]`** — compares two samplers:
   - inverse-CDF (analytical) via `θ = arccos(1 − 2u)`,
   - reject-accept on the bare PDF.
   Both are checked against the target with a histogram, an empirical
   convergence plot (`1/error²` vs sample count), and a chi-square test.

2. **Toy gamma-ray detector** — photons leave a point source 2 m above a
   1 m × 1 m planar detector. The polar angle is drawn from the sine
   distribution above (isotropic emission), the azimuth uniformly, and the
   flight path from an exponential decay PDF via reject-accept. Hits are
   binned into a 2D density plot, optionally with Gaussian smearing to mimic
   detector resolution.

3. **Signal vs background counting** — sweeps an unknown cross-section σ
   and reports the fraction of pseudo-experiments whose total count exceeds
   the background expectation. Used to locate the σ at which 95% of
   experiments would observe a positive signal.


## Run it

```bash
pip install numpy scipy matplotlib
python monte_carlo_simulation.py
```

Plots open one after another; close each window to advance.

## Files

- [monte_carlo_simulation.py](monte_carlo_simulation.py) — all three demos.

## Notes on the implementation

- The reject-accept sampler estimates the PDF supremum from a 1000-point grid
  rather than asking the caller for it; this is safe for the smooth 1D PDFs
  used here.
- Per-sample timing is approximate: the vectorised analytical sampler records
  a single elapsed time, replicated across samples, so its `times` array is
  useful only for the total cost, not the marginal cost.
- The gamma-ray example treats forward-going photons only (`θ < π/2`); the
  detector geometry discards anything that lands outside `[-1, 1]²`.
