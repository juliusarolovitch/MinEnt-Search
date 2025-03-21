#!/usr/bin/env python
# UAV dynamics model (point-mass model)

import torch

# UAV Dynamics Module
class UAVDynamics(torch.nn.Module):
    def __init__(self, start_pose, traj_steps, init_controls=None, device="cpu", v_max=1.0, omega_max=0.5, z_max=1.0):
        super(UAVDynamics, self).__init__()
        
        self.v_max = v_max  # Maximum linear speed
        self.omega_max = omega_max  # Maximum yaw rate
        self.z_max = z_max  # Maximum altitude speed

        if not isinstance(start_pose, torch.Tensor):
            start_pose = torch.tensor(start_pose, requires_grad=True)

        if init_controls is None:
            init_controls = torch.zeros((traj_steps, 4), requires_grad=True)  # vx, vy, vz, omega

        elif init_controls.shape[0] != traj_steps:
            print("[INFO] Initial controls do not have correct length, initializing to zero")
            init_controls = torch.zeros((traj_steps, 4), requires_grad=True)

        if not isinstance(init_controls, torch.Tensor):
            init_controls = torch.tensor(init_controls, requires_grad=True)

        self.device = device
        self.traj_steps = traj_steps
        self.controls = torch.nn.Parameter(init_controls)
        self.register_buffer("start_pose", start_pose)

    # Compute trajectory given controls
    def forward(self):
        # Clamp controls to enforce velocity limits
        clamped_controls = torch.clone(self.controls)
        clamped_controls[:, 0] = self.v_max * torch.tanh(clamped_controls[:, 0])  # vx
        clamped_controls[:, 1] = self.v_max * torch.tanh(clamped_controls[:, 1])  # vy
        clamped_controls[:, 2] = self.z_max * torch.tanh(clamped_controls[:, 2])  # vz
        clamped_controls[:, 3] = self.omega_max * torch.tanh(clamped_controls[:, 3])  # yaw rate

        # Integrate the velocities to get positions
        x = self.start_pose[0] + torch.cumsum(clamped_controls[:, 0], axis=0)
        y = self.start_pose[1] + torch.cumsum(clamped_controls[:, 1], axis=0)
        z = self.start_pose[2] + torch.cumsum(clamped_controls[:, 2], axis=0)
        yaw = self.start_pose[3] + torch.cumsum(clamped_controls[:, 3], axis=0)

        traj = torch.stack((x, y, z, yaw), dim=1)
        return traj

    # Compute inverse (trajectory to controls)
    def inverse(self, traj):
        if not isinstance(traj, torch.Tensor):
            traj = torch.tensor(traj, device=self.device)

        traj_with_start = torch.cat((self.start_pose.unsqueeze(0), traj), axis=0)
        traj_diff = torch.diff(traj_with_start, axis=0)

        vx = torch.clamp(traj_diff[:, 0], -self.v_max, self.v_max)
        vy = torch.clamp(traj_diff[:, 1], -self.v_max, self.v_max)
        vz = torch.clamp(traj_diff[:, 2], -self.z_max, self.z_max)
        omega = torch.clamp(traj_diff[:, 3], -self.omega_max, self.omega_max)

        controls = torch.stack((vx, vy, vz, omega), dim=1)
        return controls

# Example usage
if __name__ == "__main__":
    start_pose = [0.0, 0.0, 0.0, 0.0]  # x, y, z, yaw
    traj_steps = 100
    uav = UAVDynamics(start_pose=start_pose, traj_steps=traj_steps)
    traj = uav.forward()
    print("Generated UAV Trajectory:", traj)
