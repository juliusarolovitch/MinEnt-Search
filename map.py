import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")

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

        # Precompute the coordinate grid
        self.ygrid, self.xgrid = np.mgrid[0:dim, 0:dim]

        # Generate the map
        self.generate_map()

    def generate_map(self):
        """
        Generate the Gaussian map by randomly placing Gaussian distributions.
        """
        # Generate all random positions and parameters at once
        locs = np.random.rand(self.num_gaussians, 2) * self.dim
        
        # Generate independent standard deviations for x and y axes between 100 and 400
        stds_x = np.random.uniform(50, 200, size=self.num_gaussians)
        stds_y = np.random.uniform(50, 200, size=self.num_gaussians)
        
        # Random weights between 1 and 10 for each Gaussian
        weights = np.random.uniform(1, 10, size=self.num_gaussians)

        for i in tqdm(range(self.num_gaussians), desc="Generating map"):
            # Calculate distance from the Gaussian center for each axis
            dx = self.xgrid - locs[i, 0]
            dy = self.ygrid - locs[i, 1]

            # Direct calculation of the anisotropic Gaussian
            gaussian = np.exp(-0.5 * ((dx / stds_x[i]) ** 2 + (dy / stds_y[i]) ** 2))

            # Weight the Gaussian and add to the map
            self.map_ex += weights[i] * gaussian

        # Normalize the map so that the total probability sums to 1
        self.map_ex /= np.sum(self.map_ex)

        # Save the map as a PNG
        self.plot_map()

    def get_map(self):
        """
        Get the generated map.
        :return: Normalized Gaussian map.
        """
        return self.map_ex

    def plot_map(self, filename="curr_map.png"):
        """
        Save the generated map as a PNG file.
        """
        plt.imshow(self.map_ex, cmap='viridis', origin='lower')
        plt.colorbar(label="Intensity")
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    # Create a Gaussian map with 1000x1000 dimensions and 15 random Gaussians
    map_generator = GaussianMap(dim=2000, num_gaussians=15)
    gaussian_map = map_generator.get_map()

    # Save the map as an image
    map_generator.plot_map("generated_anisotropic_gaussians.png")
