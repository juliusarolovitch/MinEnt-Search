import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt

class GaussianMap:
    def __init__(self, dim, num_gaussians):
        """
        Initialize the Gaussian map generator.
        
        :param dim: Dimension of the map (dim x dim).
        :param num_gaussians: Number of Gaussian distributions to generate.
        """
        self.dim = dim
        self.num_gaussians = num_gaussians
        self.map_ex = np.zeros((dim, dim))
        self.res = 1 / dim
        self.ygrid, self.xgrid = np.mgrid[0:1:self.res, 0:1:self.res]
        self.map_grid = np.dstack((self.xgrid, self.ygrid))

        self.generate_map()

    def generate_map(self):
        """
        Generate the Gaussian map by randomly placing Gaussian distributions.
        """
        for _ in range(self.num_gaussians):
            # Randomly choose the mean (location) in [0, 1] x [0, 1]
            loc = np.random.rand(2)

            # Randomly choose the standard deviation.
            # Making Gaussians smaller by reducing the std range.
            std = np.random.uniform(0.005, 0.02)  # Smaller spread than before

            # Randomly choose a weight for the Gaussian
            weight = np.random.uniform(0.5, 1.5)

            # Create the Gaussian distribution
            dist = norm(loc, [std, std])
            vals = dist.pdf(self.map_grid)

            # Add to the map with the given weight
            self.map_ex += weight * vals

        # Normalize the map so that the total probability sums to 1
        self.map_ex /= np.sum(self.map_ex)

    def get_map(self):
        """
        Get the generated map.
        :return: Normalized Gaussian map.
        """
        return self.map_ex

    def plot_map(self):
        """
        Plot the generated map.
        """
        plt.imshow(self.map_ex, cmap='viridis', origin='lower')
        plt.colorbar(label="Intensity")
        plt.show()

if __name__ == "__main__":
    # Create a Gaussian map with 1000x1000 dimensions and 15 random Gaussians
    map_generator = GaussianMap(dim=1000, num_gaussians=15)
    gaussian_map = map_generator.get_map()

    # Plot the map to visualize
    map_generator.plot_map()
