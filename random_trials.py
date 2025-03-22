#!/usr/bin/env python

import argparse
import os
import cupy as cp
np = cp  # Use CuPy’s array operations (note: some functions, e.g., scipy.stats, may need adjustment)
import matplotlib.pyplot as plt
from map import GaussianMap  # Import the map function from map.py
from datetime import datetime
from tqdm import tqdm

# Utility function to calculate entropy
def entropy(prob_map):
    return -cp.sum(prob_map * cp.log(cp.clip(prob_map, 1e-12, 1)))

# Utility function to calculate KL divergence between two distributions P and Q
def kl_divergence(P, Q):
    eps = 1e-12
    # Normalize both distributions
    P_norm = P / cp.sum(P)
    Q_norm = Q / cp.sum(Q)
    return cp.sum(P_norm * cp.log(cp.clip(P_norm, eps, None) / cp.clip(Q_norm, eps, None)))

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
        new_x = np.clip(self.x + vx, 0, self.dim - 1)
        new_y = np.clip(self.y + vy, 0, self.dim - 1)
        interpolated_positions = self.get_interpolated_positions((self.x, self.y), (new_x, new_y))
        self.x, self.y = new_x, new_y
        return interpolated_positions

    def get_interpolated_positions(self, start, end, steps=10):
        """
        Generate intermediate positions between the start and end points.
        """
        x_positions = np.linspace(start[0], end[0], steps)
        y_positions = np.linspace(start[1], end[1], steps)
        return list(zip(x_positions, y_positions))

    def get_position(self):
        return self.x, self.y

    def get_detection_probability(self, x, y):
        """
        Computes detection probability for a given position.
        """
        x_min = max(0, int(x - self.sensing_radius))
        x_max = min(self.dim, int(x + self.sensing_radius) + 1)
        y_min = max(0, int(y - self.sensing_radius))
        y_max = min(self.dim, int(y + self.sensing_radius) + 1)
        
        # Create local grid for the sensing region
        local_xgrid, local_ygrid = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        dx = local_xgrid - x
        dy = local_ygrid - y
        dist = cp.sqrt(dx**2 + dy**2)
        
        # Mask for points within the sensing radius
        mask = dist <= self.sensing_radius
        sigma = self.sensing_radius / 3.034
        p_d = cp.zeros_like(dist)
        p_d[mask] = cp.exp(-0.5 * (dist[mask] / sigma) ** 2)
        return p_d, (x_min, x_max, y_min, y_max), mask

# Plot cumulative trajectories and final map if requested
def plot_trajectories(map_ex, agent_trajectories, dim, step, save_visuals):
    # Convert CuPy array to NumPy array for plotting
    map_ex_cpu = cp.asnumpy(map_ex)
    plt.imshow(map_ex_cpu, cmap='hot', origin='lower', extent=(0, dim, 0, dim))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']
    for idx, traj in enumerate(agent_trajectories):
        traj = np.array(traj)  # Trajectories are stored as CPU NumPy arrays
        plt.plot(traj[:, 0], traj[:, 1], color=colors[idx % len(colors)], label=f'Agent {idx+1}')
    plt.title(f'Cumulative Agent Trajectories at Step {step}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, dim)
    plt.ylim(0, dim)
    if save_visuals:
        plt.savefig(f'random_trials/cumulative_trajectories_step_{step}.png')
    plt.close()

# Simulation function
def simulate(dim, num_gaussians, num_quadrupeds, num_uavs, steps, save_visuals):
    if save_visuals and not os.path.exists("random_trials"):
        os.makedirs("random_trials")

    print("Generating map...")
    map_gen = GaussianMap(dim, num_gaussians)
    map_ex = map_gen.get_map()  # Assume this returns a NumPy array
    map_ex = cp.array(map_ex)   # Convert to CuPy array for GPU operations
    # Normalize map just in case
    map_ex = map_ex / cp.sum(map_ex)
    
    initial_entropy = entropy(map_ex)
    entropy_values = [initial_entropy.get()]  # Convert to CPU number for logging
    print("Map generated...")

    # Initialize the cumulative sensing coverage with a very small constant to avoid division-by-zero issues
    cumulative_coverage = cp.full(map_ex.shape, 1e-12)
    # Record initial KL divergence (at step 0) for cumulative and current sensing (both are similar at start)
    initial_kl = kl_divergence(cumulative_coverage, map_ex)
    kl_values_cumulative = [initial_kl.get()]
    kl_values_current = [initial_kl.get()]

    # Initialize agents: quadrupeds move at 2 pixels/step; UAVs at 20 pixels/step
    quadrupeds = [Agent(np.random.randint(0, dim), np.random.randint(0, dim), 2, 20, dim)
                  for _ in range(num_quadrupeds)]
    uavs = [Agent(np.random.randint(0, dim), np.random.randint(0, dim), 20, 100, dim)
             for _ in range(num_uavs)]
    agents = quadrupeds + uavs
    agent_trajectories = [[] for _ in range(len(agents))]
    print("Agents initialized...")
    
    # Set up CSV log file with datetime-stamped filename
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"random_ctrl_random_init_{current_datetime}.csv"
    with open(log_filename, "w") as log_file:
        log_file.write("Step,PercentReduction,KLdivergence_Cumulative,KLdivergence_Current\n")
        log_file.write(f"0,0.0,{initial_kl.get()},{initial_kl.get()}\n")
    print("Log file initialized...")
    
    # Simulation steps with progress bar
    for step in tqdm(range(steps), desc="Simulating..."):
        # Initialize current step's sensing distribution
        current_coverage = cp.full(map_ex.shape, 1e-12)
        
        for agent in agents:
            interpolated_positions = agent.move()
            
            # Record the agent's trajectory (store on CPU)
            agent_trajectories[agents.index(agent)].append(agent.get_position())
            
            # Update map coverage, cumulative sensing, and current sensing for each interpolated position
            for pos in interpolated_positions:
                p_d, (x_min, x_max, y_min, y_max), mask = agent.get_detection_probability(pos[0], pos[1])
                
                # Update cumulative sensing distribution
                local_cumulative = cumulative_coverage[y_min:y_max, x_min:x_max]
                local_cumulative[mask] += p_d[mask]
                cumulative_coverage[y_min:y_max, x_min:x_max] = local_cumulative
                
                # Update current step's sensing distribution
                local_current = current_coverage[y_min:y_max, x_min:x_max]
                local_current[mask] += p_d[mask]
                current_coverage[y_min:y_max, x_min:x_max] = local_current
                
                # Update the map_ex using the same sensing probabilities
                local_region = map_ex[y_min:y_max, x_min:x_max]
                prior = local_region[mask]
                updated = (prior * (1 - p_d[mask])) / (prior * (1 - p_d[mask]) + (1 - prior) + 1e-12)
                local_region[mask] = updated
                map_ex[y_min:y_max, x_min:x_max] = local_region
            
            # Renormalize the entire map after each agent's updates
            map_ex /= cp.sum(map_ex)

        # Compute and record the metrics after processing all agents for this step
        current_entropy = entropy(map_ex)
        entropy_values.append(current_entropy.get())
        percent_reduction = ((initial_entropy - current_entropy) / initial_entropy) * 100
        percent_reduction = percent_reduction.get()
        
        # Compute KL divergence between the normalized cumulative sensing distribution and the current information map
        current_kl_cumulative = kl_divergence(cumulative_coverage, map_ex)
        kl_values_cumulative.append(current_kl_cumulative.get())
        
        # Compute KL divergence for the current step's sensing distribution against the current map
        current_kl_current = kl_divergence(current_coverage, map_ex)
        kl_values_current.append(current_kl_current.get())
        
        print(f"Step {step + 1} Percent Entropy Reduction: {percent_reduction:.4f}%, KL Cumulative: {current_kl_cumulative.get():.6f}, KL Current: {current_kl_current.get():.6f}")
        with open(log_filename, "a") as log_file:
            log_file.write(f"{step+1},{percent_reduction},{current_kl_cumulative.get()},{current_kl_current.get()}\n")

    # Plot entropy reduction over time
    plt.figure()
    plt.plot(entropy_values, marker='o', linestyle='-', color='blue')
    plt.title("Entropy Reduction Over Time")
    plt.xlabel("Global Trajectory Step")
    plt.ylabel("Entropy")
    plt.grid(True)
    if save_visuals:
        plt.savefig('random_trials/entropy_reduction.png')
    plt.show()

    # Plot cumulative KL divergence over time
    plt.figure()
    plt.plot(kl_values_cumulative, marker='o', linestyle='-', color='green')
    plt.title("Cumulative KL Divergence Over Time")
    plt.xlabel("Global Trajectory Step")
    plt.ylabel("KL Divergence")
    plt.grid(True)
    if save_visuals:
        plt.savefig('random_trials/kl_divergence_cumulative.png')
    plt.show()

    # Plot current step KL divergence over time
    plt.figure()
    plt.plot(kl_values_current, marker='o', linestyle='-', color='magenta')
    plt.title("Current Step KL Divergence Over Time")
    plt.xlabel("Global Trajectory Step")
    plt.ylabel("KL Divergence")
    plt.grid(True)
    if save_visuals:
        plt.savefig('random_trials/kl_divergence_current.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Entropy Reduction Simulation")
    parser.add_argument("--dim", type=int, default=2000, help="Map dimension (dim x dim)")
    parser.add_argument("--num_quadrupeds", type=int, default=10, help="Number of quadruped agents")
    parser.add_argument("--num_uavs", type=int, default=8, help="Number of UAV agents")
    parser.add_argument("--steps", type=int, default=7200, help="Number of global trajectory steps")
    parser.add_argument("--save_visuals", action="store_true", help="Save trajectory and metric visualizations to files")
    args = parser.parse_args()

    simulate(args.dim, 15, args.num_quadrupeds, args.num_uavs, args.steps, args.save_visuals)
