#!/usr/bin/env python

import argparse
import os
import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from map import GaussianMap  # Import the map function from map.py
from datetime import datetime

# Utility function to calculate entropy
def entropy(prob_map):
    return -np.sum(prob_map * np.log(np.clip(prob_map, 1e-12, 1)))

# Agent class for both quadrupeds and UAVs
class Agent:
    def __init__(self, x, y, speed, sensing_radius, dim):
        self.x = x
        self.y = y
        self.speed = speed
        self.sensing_radius = sensing_radius  # This is the 99% confidence radius
        self.dim = dim

    def move(self):
        vx = np.random.uniform(-self.speed, self.speed)
        vy = np.random.uniform(-self.speed, self.speed)
        self.x = np.clip(self.x + vx, 0, self.dim - 1)
        self.y = np.clip(self.y + vy, 0, self.dim - 1)

    def get_position(self):
        return self.x, self.y

    def get_detection_probability(self, dim):
        """
        Returns:
          - p_d: the detection probability function over the map (2D array)
          - mask: a boolean mask for points within the 99% confidence region.
        """
        xgrid, ygrid = np.meshgrid(np.arange(dim), np.arange(dim))
        dx = xgrid - self.x
        dy = ygrid - self.y
        dist = np.sqrt(dx**2 + dy**2)
        # Mask for points within the 99% confidence range
        mask = dist <= self.sensing_radius
        # Convert the provided sensing_radius (99% range) to sigma:
        sigma = self.sensing_radius / 3.034
        p_d = np.zeros_like(dist)
        # Gaussian detection probability that peaks at 1
        p_d[mask] = np.exp(-0.5 * (dist[mask] / sigma) ** 2)
        return p_d, mask

# Plot cumulative trajectories and final map if requested
def plot_trajectories(map_ex, agent_trajectories, dim, step, save_visuals):
    plt.imshow(map_ex, cmap='hot', origin='lower', extent=(0, dim, 0, dim))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']
    for idx, traj in enumerate(agent_trajectories):
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], color=colors[idx % len(colors)], label=f'Agent {idx+1}')
    plt.title(f'Cumulative Agent Trajectories at Step {step}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, dim)
    plt.ylim(0, dim)
    if save_visuals:
        plt.savefig(f'random_trials/cumulative_trajectories_step_{step}.png')
    # plt.show()

# Simulation function
def simulate(dim, num_gaussians, num_quadrupeds, num_uavs, steps, save_visuals):
    # Create directory for saving visuals if needed
    if save_visuals and not os.path.exists("random_trials"):
        os.makedirs("random_trials")

    # Generate the map (which is now a normalized probability distribution)
    map_gen = GaussianMap(dim, num_gaussians)
    map_ex = map_gen.get_map()
    initial_entropy = entropy(map_ex)
    entropy_values = [initial_entropy]

    # Initialize agents: quadrupeds move at 1.5 pixels/step; UAVs at 20 pixels/step
    quadrupeds = [Agent(np.random.randint(0, dim), np.random.randint(0, dim), 1.5, 20, dim) for _ in range(num_quadrupeds)]
    uavs = [Agent(np.random.randint(0, dim), np.random.randint(0, dim), 20, 100, dim) for _ in range(num_uavs)]
    agents = quadrupeds + uavs
    agent_trajectories = [[] for _ in range(len(agents))]
    
    # Set up entropy log file with a datetime-stamped filename
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"random_ctrl_random_init_{current_datetime}.txt"
    with open(log_filename, "w") as log_file:
        log_file.write("Step,Entropy\n")
        log_file.write(f"0,{initial_entropy}\n")
    
    # Simulation steps
    for step in range(steps):
        for agent in agents:
            agent.move()
            # Record the agent's trajectory
            agent_trajectories[agents.index(agent)].append(agent.get_position())
            
            # Get the detection probability function and mask for the agent's sensor
            p_d, mask = agent.get_detection_probability(dim)
            prior = map_ex[mask]
            # Bayesian update for each pixel in the sensor's 99% confidence region
            updated = (prior * (1 - p_d[mask])) / (prior * (1 - p_d[mask]) + (1 - prior) + 1e-12)
            map_ex[mask] = updated

            # Renormalize the map so that it remains a proper probability distribution
            map_ex /= np.sum(map_ex)

        current_entropy = entropy(map_ex)
        entropy_values.append(current_entropy)
        print(f"Step {step + 1} Entropy: {current_entropy:.4f} (Reduction: {initial_entropy - current_entropy:.4f})")
        
        # Append the current entropy value to the log file
        with open(log_filename, "a") as log_file:
            log_file.write(f"{step+1},{current_entropy}\n")
        
        # Plot trajectories at each step if saving visuals
        plot_trajectories(map_ex, agent_trajectories, dim, step + 1, save_visuals)

    # Plot the overall entropy reduction
    plt.plot(entropy_values, marker='o', linestyle='-', color='blue')
    plt.title("Entropy Reduction Over Time")
    plt.xlabel("Global Trajectory Step")
    plt.ylabel("Entropy")
    plt.grid(True)
    if save_visuals:
        plt.savefig('random_trials/entropy_reduction.png')
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Entropy Reduction Simulation")
    parser.add_argument("--dim", type=int, default=100, help="Map dimension (dim x dim)")
    parser.add_argument("--num_quadrupeds", type=int, default=5, help="Number of quadruped agents")
    parser.add_argument("--num_uavs", type=int, default=5, help="Number of UAV agents")
    parser.add_argument("--steps", type=int, default=50, help="Number of global trajectory steps")
    parser.add_argument("--save_visuals", action="store_true", help="Save trajectory and entropy visualizations to files")
    args = parser.parse_args()

    simulate(args.dim, 10, args.num_quadrupeds, args.num_uavs, args.steps, args.save_visuals)
