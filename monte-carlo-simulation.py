import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.stats import chisquare

PI = np.pi
BACKGROUND_MEAN = 5.8

class DistributionMethods:
    """Base class for distribution methods."""

    def __init__(self):
        pass

    def analytical(self, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate samples using the analytical method."""
        tic = time.time()
        theta = np.arccos(np.random.uniform(-1.0, 1.0, num_samples))
        toc = np.array([time.time() - tic for _ in range(num_samples)])
        return theta, toc

    def reject_accept(self, pdf: callable, num_samples: int, xmin: float, xmax: float) -> tuple[np.ndarray, np.ndarray]:
        """Generate samples using the reject-accept method."""
        tic = time.time()
        accepted = []
        toc = []
        max_iterations = 1000000  # Set a maximum number of iterations to avoid infinite loops
        iteration = 0
        while len(accepted) < num_samples and iteration < max_iterations:
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(0, pdf(x).max())
            if y < pdf(x):
                accepted.append(x)
                toc.append(time.time() - tic)
            iteration += 1
        return np.array(accepted), np.array(toc)

class SineDistribution(DistributionMethods):
    """Class for generating and analyzing sine distribution."""

    def __init__(self):
        super().__init__()

    def plot_histogram(self, num_samples: int, method: str):
        """Plot the histogram of the normalized frequency of theta."""
        theta = np.linspace(0, PI, num_samples)
        if method == "analytical":
            observed, _ = self.analytical(num_samples)
            label = "Analytical method"
        elif method == "reject_accept":
            observed, _ = self.reject_accept(np.sin, num_samples, 0.0, PI)
            label = "Reject-Accept"
        else:
            raise ValueError(f"Invalid method: {method}")

        plt.hist(observed, bins="auto", density=True, alpha=0.3, histtype="stepfilled", color="steelblue", edgecolor="none", label=label)
        plt.plot(theta, 0.5 * np.sin(theta), label="Sine function", color="red")
        plt.title("Histogram of the normalized frequency of theta, overlaid with a sin(theta) function.")
        plt.xlabel("Theta")
        plt.ylabel("Normalized Frequency")
        plt.legend(loc="upper right")
        plt.show()

    def calculate_error(self, num_samples: int) -> tuple[float, float]:
        """Calculate the error between observed and expected frequencies."""
        observed_analytical, _ = self.analytical(num_samples)
        observed_reject_accept, _ = self.reject_accept(np.sin, num_samples, 0.0, PI)

        n_analytical, bins_analytical = np.histogram(observed_analytical, bins="auto", density=True)
        n_reject_accept, bins_reject_accept = np.histogram(observed_reject_accept, bins="auto", density=True)

        expected_analytical = 0.5 * np.sin(bins_analytical[:-1])
        expected_reject_accept = 0.5 * np.sin(bins_reject_accept[:-1])

        error_analytical = np.sqrt(np.sum((n_analytical - expected_analytical) ** 2))
        error_reject_accept = np.sqrt(np.sum((n_reject_accept - expected_reject_accept) ** 2))

        return error_analytical, error_reject_accept

    def plot_error(self, num_samples_range: np.ndarray):
        """Plot the inverse of absolute error squared as a function of the number of samples."""
        error_sum_analytical = np.zeros(len(num_samples_range))
        error_sum_reject_accept = np.zeros(len(num_samples_range))

        for i, num_samples in enumerate(num_samples_range):
            error_sum_analytical[i], error_sum_reject_accept[i] = self.calculate_error(num_samples)

        plt.plot(num_samples_range, 1 / (error_sum_analytical ** 2), marker="o", label="Analytical")
        plt.plot(num_samples_range, 1 / (error_sum_reject_accept ** 2), marker="o", label="Reject-Accept")
        plt.xlabel("Samples")
        plt.ylabel("Inverse of absolute error squared")
        plt.legend(loc="upper right")
        plt.title("Inverse of absolute error squared as a function of the number of samples.")
        plt.show()

    def plot_time(self, num_samples: int):
        """Plot the time taken as a function of the number of samples."""
        _, time_analytical = self.analytical(num_samples)
        _, time_reject_accept = self.reject_accept(np.sin, num_samples, 0.0, PI)

        plt.plot(np.arange(len(time_analytical)), time_analytical, label="Analytical")
        plt.plot(np.arange(len(time_reject_accept)), time_reject_accept, label="Reject-accept")
        plt.title("Time taken as a function of the number of samples, N.")
        plt.xlabel("Samples")
        plt.ylabel("Time (s)")
        plt.legend(loc="upper right")
        plt.show()

    def calculate_chi_square(self, num_samples: int):
        """Calculate and print the chi-square p-value for both methods."""
        theta_analytical, _ = self.analytical(num_samples)
        theta_reject_accept, _ = self.reject_accept(np.sin, num_samples, 0.0, PI)

        obs_freq_analytical, bin_edges_analytical = np.histogram(theta_analytical, bins="auto", density=True)
        obs_freq_reject_accept, bin_edges_reject_accept = np.histogram(theta_reject_accept, bins="auto", density=True)

        exp_freq_analytical = 0.5 * np.sin((bin_edges_analytical[:-1] + bin_edges_analytical[1:]) / 2)
        exp_freq_reject_accept = 0.5 * np.sin((bin_edges_reject_accept[:-1] + bin_edges_reject_accept[1:]) / 2)

        _, p_analytical = chisquare(obs_freq_analytical, exp_freq_analytical)
        _, p_reject_accept = chisquare(obs_freq_reject_accept, exp_freq_reject_accept)

        print(f"P-value for the analytical method at N = {num_samples} is P = {p_analytical}")
        print(f"P-value for the reject-accept method at N = {num_samples} is P = {p_reject_accept}")

class GammaRayDetection(DistributionMethods):
    """Class for simulating and analyzing gamma ray detection."""

    def __init__(self, num_samples: int, smear: bool):
        super().__init__()
        self.num_samples = num_samples
        self.smear = smear

    def decay(self, x: float) -> float:
        """Decay function."""
        return np.exp(-1.04 * x)

    def calculate_detection_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the coordinates of detected gamma rays."""
        thetas, _ = self.analytical(self.num_samples)
        phis = np.random.uniform(0, 2 * PI, self.num_samples)
        randdist, _ = self.reject_accept(self.decay, self.num_samples, 0, 2)
        det_dist = 2 - randdist

        x_coords = np.array([])
        y_coords = np.array([])

        for i in range(self.num_samples):
            if det_dist[i] > 0 and thetas[i] < PI / 2:
                r = det_dist[i] / np.cos(thetas[i])
                new_x = r * np.sin(thetas[i]) * np.cos(phis[i])
                new_y = r * np.sin(thetas[i]) * np.sin(phis[i])

                if abs(new_x) <= 1 and abs(new_y) <= 1:
                    if self.smear:
                        smear_x = np.random.normal(new_x, 0.1 / 3)
                        smear_y = np.random.normal(new_y, 0.3 / 3)
                        if abs(smear_x) < 5 and abs(smear_y) < 5:
                            x_coords = np.append(x_coords, smear_x)
                            y_coords = np.append(y_coords, smear_y)
                    else:
                        x_coords = np.append(x_coords, new_x)
                        y_coords = np.append(y_coords, new_y)
        return x_coords, y_coords

    def plot_coordinates_distribution(self):
        """Plot the density of detected gamma rays at given coordinates."""
        x_coords, y_coords = self.calculate_detection_coordinates()
        plt.hist2d(x_coords, y_coords, bins=(50, 50), cmap=plt.cm.jet)
        plt.colorbar()
        plt.xlabel("x(m)")
        plt.ylabel("y(m)")
        plt.title("Density plot of the distribution of gamma rays detected at a given coordinate (x,y).")
        plt.show()

    def plot_spherical_distribution(self):
        """Plot the uniform distribution of points on a sphere."""
        thetas, _ = self.analytical(self.num_samples)
        phis = np.random.uniform(0, 2 * PI, self.num_samples)

        x = 2 * np.sin(thetas) * np.cos(phis)
        y = 2 * np.sin(thetas) * np.sin(phis)
        z = 2 * np.cos(thetas)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, c="blue", alpha=0.3)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("Uniform distribution of points on a sphere")
        plt.show()

    def plot_inverse_square_law(self):
        """Plot the inverse of the root of the frequency against distance."""
        x_coords, _ = self.calculate_detection_coordinates()
        x_pixel = np.linspace(-1, 1, 200)
        density_x = np.zeros(len(x_pixel) - 1)

        for i in range(len(x_pixel) - 1):
            density_x[i] = np.sum((x_coords > x_pixel[i]) & (x_coords <= x_pixel[i + 1]))

        plt.plot(x_pixel[:-1], np.sqrt(1 / density_x), label="Inverse Square Law")
        plt.xlabel("x(m)")
        plt.ylabel("Inverse of root of frequency")
        plt.title("Inverse of the root of the frequency against distance")
        plt.legend()
        plt.show()

class SignalDetection(DistributionMethods):
    """Class for simulating and analyzing signal detection."""

    def __init__(self):
        super().__init__()

    def calculate_total_count(self, cross_section: float, num_samples: int) -> np.ndarray:
        """Calculate the total count for a given cross-section."""
        background_mean = np.random.normal(BACKGROUND_MEAN, 0.4, num_samples)
        luminosity = np.random.normal(10, 0.5, num_samples)

        background_count = np.random.poisson(background_mean)
        signal_count = np.random.poisson(cross_section * luminosity)

        return background_count + signal_count

    def plot_confidence(self, num_samples: int):
        """Plot the confidence level of values above 5 counts for a given cross-section."""
        cross_section_range = np.linspace(0, 0.05, num_samples)
        confidences = []

        for cross_section in cross_section_range:
            total_counts = self.calculate_total_count(cross_section, num_samples)
            percent_above_five = np.sum(total_counts > 5) / num_samples
            confidences.append(percent_above_five)

            if percent_above_five >= 0.95:
                break

        plt.plot(cross_section_range[:len(confidences)], confidences)
        plt.xlabel("Cross-section (sigma)")
        plt.ylabel("Confidence")
        plt.title("Confidence level of values above 5 counts for a given cross-section")
        plt.show()

        if len(confidences) < len(cross_section_range):
            print(f"Cross-section at which the total counts are 95% above 5 = {cross_section_range[len(confidences)-1]:.4f}")
        else:
            print("95% confidence not reached within the given cross-section range.")