"""Monte Carlo studies of three small physics problems.

1. Drawing samples from ``p(theta) = sin(theta)/2`` on ``[0, pi]`` using both
   an analytical (inverse-CDF) method and a reject-accept method, and
   comparing convergence and runtime.
2. A toy gamma-ray detector: photons leave a point source with isotropic
   direction and an exponentially distributed flight path, and we look at
   the resulting hit distribution on a planar detector, optionally with
   Gaussian smearing.
3. A signal-vs-background counting experiment, finding the smallest
   cross-section at which 95% of pseudo-experiments produce a count above
   the background-only expectation.

Run ``python monte_carlo_simulation.py`` to execute all three demos.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKGROUND_MEAN = 5.8
DECAY_RATE = 1.04           # exponential decay constant for the gamma ray demo
DETECTOR_HALF_WIDTH = 1.0   # planar detector spans [-1, 1] in x and y
SOURCE_OFFSET = 2.0         # source sits 2 units above the detector


# ---------------------------------------------------------------------------
# Sampling primitives
# ---------------------------------------------------------------------------


@dataclass
class TimedSamples:
    """Samples paired with the elapsed time at the moment each was accepted."""
    samples: np.ndarray
    times: np.ndarray


def sample_sine_analytical(n: int) -> TimedSamples:
    """Sample ``p(theta) = sin(theta)/2`` via inverse CDF (`arccos(1-2u)`)."""
    start = time.perf_counter()
    samples = np.arccos(np.random.uniform(-1.0, 1.0, n))
    elapsed = time.perf_counter() - start
    # Vectorised draw: each sample shares the same elapsed time.
    return TimedSamples(samples, np.full(n, elapsed))


def sample_reject_accept(
    pdf: Callable[[np.ndarray], np.ndarray],
    n: int,
    xmin: float,
    xmax: float,
    pdf_max: float | None = None,
    max_iterations: int = 1_000_000,
) -> TimedSamples:
    """Sample from ``pdf`` on ``[xmin, xmax]`` using rejection sampling.

    ``pdf_max`` is an upper bound on ``pdf`` over the interval. If omitted we
    estimate it from a dense grid (cheap for the 1D cases used here).
    """
    if pdf_max is None:
        grid = np.linspace(xmin, xmax, 1000)
        pdf_max = float(np.max(pdf(grid)))

    accepted: list[float] = []
    times: list[float] = []

    start = time.perf_counter()
    for _ in range(max_iterations):
        if len(accepted) >= n:
            break
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(0.0, pdf_max)
        if y < pdf(np.array([x]))[0]:
            accepted.append(x)
            times.append(time.perf_counter() - start)

    return TimedSamples(np.array(accepted), np.array(times))


# ---------------------------------------------------------------------------
# Demo 1: sampling sin(theta) / 2 on [0, pi]
# ---------------------------------------------------------------------------


def _sine_pdf(theta: np.ndarray) -> np.ndarray:
    return np.sin(theta)


def plot_sine_histogram(n: int, method: str) -> None:
    if method == "analytical":
        observed = sample_sine_analytical(n).samples
        label = "Analytical (inverse CDF)"
    elif method == "reject_accept":
        observed = sample_reject_accept(_sine_pdf, n, 0.0, np.pi, pdf_max=1.0).samples
        label = "Reject-accept"
    else:
        raise ValueError(f"Unknown method: {method!r}")

    theta = np.linspace(0, np.pi, 500)
    plt.hist(
        observed,
        bins="auto",
        density=True,
        alpha=0.3,
        histtype="stepfilled",
        color="steelblue",
        label=label,
    )
    plt.plot(theta, 0.5 * np.sin(theta), color="red", label="sin(θ)/2")
    plt.title("Sampled distribution vs target sin(θ)/2")
    plt.xlabel("θ")
    plt.ylabel("Normalised frequency")
    plt.legend(loc="upper right")
    plt.show()


def _empirical_error(samples: np.ndarray) -> float:
    """RMS deviation between histogram density and the target sin(θ)/2."""
    density, edges = np.histogram(samples, bins="auto", density=True)
    expected = 0.5 * np.sin(edges[:-1])
    return float(np.sqrt(np.sum((density - expected) ** 2)))


def plot_sine_convergence(sample_sizes: np.ndarray) -> None:
    inv_sq_analytical = np.empty(len(sample_sizes))
    inv_sq_reject = np.empty(len(sample_sizes))

    for i, n in enumerate(sample_sizes):
        analytical = sample_sine_analytical(n).samples
        reject = sample_reject_accept(_sine_pdf, n, 0.0, np.pi, pdf_max=1.0).samples
        inv_sq_analytical[i] = 1.0 / _empirical_error(analytical) ** 2
        inv_sq_reject[i] = 1.0 / _empirical_error(reject) ** 2

    plt.plot(sample_sizes, inv_sq_analytical, marker="o", label="Analytical")
    plt.plot(sample_sizes, inv_sq_reject, marker="o", label="Reject-accept")
    plt.xlabel("Samples")
    plt.ylabel("1 / error²")
    plt.title("Convergence rate of the two samplers")
    plt.legend(loc="upper right")
    plt.show()


def report_chi_square(n: int) -> None:
    analytical = sample_sine_analytical(n).samples
    reject = sample_reject_accept(_sine_pdf, n, 0.0, np.pi, pdf_max=1.0).samples

    for label, observed in (("analytical", analytical), ("reject-accept", reject)):
        density, edges = np.histogram(observed, bins="auto", density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        expected = 0.5 * np.sin(centres)
        _, p_value = chisquare(density, expected)
        print(f"  {label:14s}  N={n}  p={p_value:.4f}")


# ---------------------------------------------------------------------------
# Demo 2: gamma-ray detector
# ---------------------------------------------------------------------------


def _decay_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-DECAY_RATE * x)


def gamma_ray_hits(n: int, smear: bool) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``n`` photons and return ``(x, y)`` for the ones that hit."""
    thetas = sample_sine_analytical(n).samples
    phis = np.random.uniform(0.0, 2 * np.pi, n)
    travel = sample_reject_accept(_decay_pdf, n, 0.0, SOURCE_OFFSET).samples
    distance_to_plane = SOURCE_OFFSET - travel

    forward = (distance_to_plane > 0) & (thetas < np.pi / 2)
    r = distance_to_plane[forward] / np.cos(thetas[forward])
    x = r * np.sin(thetas[forward]) * np.cos(phis[forward])
    y = r * np.sin(thetas[forward]) * np.sin(phis[forward])

    on_detector = (np.abs(x) <= DETECTOR_HALF_WIDTH) & (np.abs(y) <= DETECTOR_HALF_WIDTH)
    x, y = x[on_detector], y[on_detector]

    if smear:
        x = np.random.normal(x, 0.1 / 3)
        y = np.random.normal(y, 0.3 / 3)

    return x, y


def plot_detector_density(n: int, smear: bool = False) -> None:
    x, y = gamma_ray_hits(n, smear)
    plt.hist2d(x, y, bins=(50, 50), cmap=plt.cm.jet)
    plt.colorbar(label="Counts")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    title = "Gamma-ray hit density on detector"
    plt.title(title + (" (with Gaussian smearing)" if smear else ""))
    plt.show()


def plot_source_sphere(n: int) -> None:
    thetas = sample_sine_analytical(n).samples
    phis = np.random.uniform(0.0, 2 * np.pi, n)

    x = SOURCE_OFFSET * np.sin(thetas) * np.cos(phis)
    y = SOURCE_OFFSET * np.sin(thetas) * np.sin(phis)
    z = SOURCE_OFFSET * np.cos(thetas)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c="blue", alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Isotropic emission directions on a sphere")
    plt.show()


# ---------------------------------------------------------------------------
# Demo 3: signal-vs-background counting experiment
# ---------------------------------------------------------------------------


def simulate_total_count(cross_section: float, n: int) -> np.ndarray:
    """One pseudo-experiment per sample: ``B ~ Poisson(N(5.8,0.4))`` + signal."""
    background_mean = np.random.normal(BACKGROUND_MEAN, 0.4, n)
    luminosity = np.random.normal(10.0, 0.5, n)
    background_count = np.random.poisson(background_mean)
    signal_count = np.random.poisson(cross_section * luminosity)
    return background_count + signal_count


def plot_confidence_curve(n: int) -> None:
    """Sweep cross-section and plot fraction of pseudo-experiments above 5 counts."""
    cross_sections = np.linspace(0.0, 0.05, n)
    confidences: list[float] = []

    for sigma in cross_sections:
        counts = simulate_total_count(sigma, n)
        fraction = float(np.mean(counts > 5))
        confidences.append(fraction)
        if fraction >= 0.95:
            break

    sweep = cross_sections[: len(confidences)]
    plt.plot(sweep, confidences)
    plt.xlabel("Cross-section σ")
    plt.ylabel("P(count > 5)")
    plt.title("Discovery confidence vs cross-section")
    plt.show()

    if confidences[-1] >= 0.95:
        print(f"  95% confidence reached at σ ≈ {sweep[-1]:.4f}")
    else:
        print("  95% confidence not reached within sweep range.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    print("Demo 1: sin(θ)/2 sampler")
    plot_sine_histogram(10_000, "analytical")
    plot_sine_histogram(10_000, "reject_accept")
    plot_sine_convergence(np.array([100, 500, 1_000, 5_000, 10_000]))
    report_chi_square(10_000)

    print("\nDemo 2: gamma-ray detector")
    plot_source_sphere(2_000)
    plot_detector_density(50_000, smear=False)
    plot_detector_density(50_000, smear=True)

    print("\nDemo 3: discovery confidence sweep")
    plot_confidence_curve(500)


if __name__ == "__main__":
    main()
