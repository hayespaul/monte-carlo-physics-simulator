import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.stats import chisquare

class DistributionMethods:
    def __init__(self):
        pass

    def analytical(self, N):
        tic = time.time()
        theta = np.arccos(np.random.uniform(-1.0, 1.0, N))
        toc = np.array([time.time() - tic for _ in range(N)])
        return theta, toc

    def reject_accept(self, pdf, N, xmin, xmax):
        tic = time.time()
        accepted = []
        toc = []
        while len(accepted) < N:
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(0, pdf(x).max())
            if y < pdf(x):
                accepted.append(x)
                toc.append(time.time() - tic)
        return np.array(accepted), np.array(toc)

class Task1(DistributionMethods):
    def __init__(self):
        super().__init__()

    def hist_t1(self, N, option):
        theta = np.linspace(0, np.pi, N)
        if option == 1:
            observed, _ = self.analytical(N)
            label = 'Analytical method'
        elif option == 0:
            observed, _ = self.reject_accept(np.sin, N, 0.0, np.pi)
            label = 'Reject-Accept'
        plt.hist(observed, bins='auto', density=True, alpha=0.3, histtype='stepfilled', color='steelblue', edgecolor='none', label=label)
        plt.plot(theta, 0.5 * np.sin(theta), label='Sine function', color='red')
        plt.title('Histogram of the normalized frequency of theta, overlaid with a sin(theta) function.')
        plt.xlabel('Theta')
        plt.ylabel('Normalized Frequency')
        plt.legend(loc='upper right')
        plt.show()

    def error(self, N):
        observed, _ = self.analytical(N)
        observed1, _ = self.reject_accept(np.sin, N, 0.0, np.pi)

        n, bins = np.histogram(observed, bins='auto', density=True)
        n2, bins2 = np.histogram(observed1, bins='auto', density=True)

        expected = 0.5 * np.sin(bins[:-1])
        expected1 = 0.5 * np.sin(bins2[:-1])

        error = (n - expected) ** 2
        error1 = (n2 - expected1) ** 2

        error_sum = np.sqrt(np.sum(error))
        error_sum1 = np.sqrt(np.sum(error1))

        return error_sum, error_sum1

    def error_plot(self, N_range):
        error_sum = np.zeros(len(N_range))
        error_sum1 = np.zeros(len(N_range))

        for i, N in enumerate(N_range):
            error_sum[i], error_sum1[i] = self.error(N)

        plt.plot(N_range, 1 / (error_sum ** 2), marker='o', label='Analytical')
        plt.plot(N_range, 1 / (error_sum1 ** 2), marker='o', label='Reject-Accept')
        plt.xlabel('Samples')
        plt.ylabel('Inverse of absolute error squared')
        plt.legend(loc='upper right')
        plt.title('Inverse of absolute error squared as a function of the number of samples.')
        plt.show()

    def time_task1(self, N):
        _, ta = self.analytical(N)
        _, tr = self.reject_accept(np.sin, N, 0.0, np.pi)

        plt.plot(np.arange(len(ta)), ta, label='Analytical')
        plt.plot(np.arange(len(tr)), tr, label='Reject-accept')
        plt.title('Time taken as a function of the number of samples, N.')
        plt.xlabel('Samples')
        plt.ylabel('Time (s)')
        plt.legend(loc='upper right')
        plt.show()

    def chi_square(self, N):
        theta_a, _ = self.analytical(N)
        theta_r, _ = self.reject_accept(np.sin, N, 0.0, np.pi)

        obs_freq_a, bin_edges_a = np.histogram(theta_a, bins='auto', density=True)
        obs_freq_r, bin_edges_r = np.histogram(theta_r, bins='auto', density=True)

        exp_freq_a = 0.5 * np.sin((bin_edges_a[:-1] + bin_edges_a[1:]) / 2)
        exp_freq_r = 0.5 * np.sin((bin_edges_r[:-1] + bin_edges_r[1:]) / 2)

        _, p_a = chisquare(obs_freq_a, exp_freq_a)
        _, p_r = chisquare(obs_freq_r, exp_freq_r)

        print(f"P-value for the analytical method at N = {N} is P = {p_a}")
        print(f"P-value for the reject-accept method at N = {N} is P = {p_r}")

class PhysicsProblem1(DistributionMethods):
    def __init__(self, N, smear):
        super().__init__()
        self.N = N
        self.smear = smear

    def decay(self, x):
        return np.exp(-1.04 * x)

    def coordinates_at_detection(self):
        thetas, _ = self.analytical(self.N)
        phis = np.random.uniform(0, 2 * np.pi, self.N)
        randdist, _ = self.reject_accept(self.decay, self.N, 0, 2)
        det_dist = 2 - randdist

        x_coords = np.array([])
        y_coords = np.array([])

        for i in range(self.N):
            if det_dist[i] > 0 and thetas[i] < np.pi / 2:
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

    def distribution_coords(self):
        x_coords, y_coords = self.coordinates_at_detection()
        plt.hist2d(x_coords, y_coords, bins=(50, 50), cmap=plt.cm.jet)
        plt.colorbar()
        plt.xlabel('x(m)')
        plt.ylabel('y(m)')
        plt.title('Density plot of the distribution of gamma rays detected at a given coordinate (x,y).')
        plt.show()

    def test_spherical_dist(self):
        thetas, _ = self.analytical(self.N)
        phis = np.random.uniform(0, 2 * np.pi, self.N)

        x = 2 * np.sin(thetas) * np.cos(phis)
        y = 2 * np.sin(thetas) * np.sin(phis)
        z = 2 * np.cos(thetas)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='blue', alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Uniform distribution of points on a sphere')
        plt.show()

    def inverse_sq(self):
        x_coords, _ = self.coordinates_at_detection()
        x_pixel = np.linspace(-1, 1, 200)
        density_x = np.zeros(len(x_pixel) - 1)

        for i in range(len(x_pixel) - 1):
            density_x[i] = np.sum((x_coords > x_pixel[i]) & (x_coords <= x_pixel[i + 1]))

        plt.plot(x_pixel[:-1], np.sqrt(1 / density_x), label='Inverse Square Law')
        plt.xlabel('x(m)')
        plt.ylabel('Inverse of root of frequency')
        plt.title('Inverse of the root of the frequency against distance')
        plt.legend()
        plt.show()

class PhysicsProblem2(DistributionMethods):
    def __init__(self):
        super().__init__()

    def total_count(self, sigma, N):
        background_mean = 5.8
        mean_background = np.random.normal(background_mean, 0.4, N)
        luminosity = np.random.normal(10, 0.5, N)

        background_count = np.random.poisson(mean_background)
        signal_count = np.random.poisson(sigma * luminosity)

        return background_count + signal_count

    def confidence(self, N):
        sigma_range = np.linspace(0, 0.05, N)
        confidences = []

        for sigma in sigma_range:
            total_counts = self.total_count(sigma, N)
            percent_above_five = np.sum(total_counts > 5) / N
            confidences.append(percent_above_five)

            if percent_above_five >= 0.95:
                break

        plt.plot(sigma_range[:len(confidences)], confidences)
        plt.xlabel('Cross-section (sigma)')
        plt.ylabel('Confidence')
        plt.title('Confidence level of values above 5 counts for a given cross section')
        plt.show()

        if len(confidences) < len(sigma_range):
            print(f"Cross section at which the total counts are 95% above 5 = {sigma_range[len(confidences)-1]:.4f}")
        else:
            print("95% confidence not reached within the given sigma range.")