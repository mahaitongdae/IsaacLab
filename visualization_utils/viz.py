import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(15, 6))
tasks = ["Diagonal", "Quadratic", "OOD"]
for i, task in enumerate(tasks):
    folder = f"/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/{task}/"
    print(folder)
    ref_traj = np.loadtxt(f"{folder}trajectory.txt",
                 delimiter=" ", dtype=float)

    sac_agent = np.loadtxt(f"{folder}sac_agent_positions.txt",
                 delimiter=" ", dtype=float)
    
    ctrl_agent = np.loadtxt(f"{folder}ctrl_agent_positions.txt",
                delimiter=" ", dtype=float)
    
    print(ref_traj.shape)
    print(sac_agent.shape)
    print(ctrl_agent.shape)
    
    axs[i].scatter(ref_traj[:250, 0], ref_traj[:250, 1], s= 50, alpha=1, label='Reference Trajectories')
    axs[i].scatter(sac_agent[:500, 0], sac_agent[:500, 1], marker='x',alpha=0.3, s=40, label='SAC Multi-Traj')
    axs[i].scatter(ctrl_agent[:500, 0], ctrl_agent[:500, 1], marker='x',alpha=0.3, s=40, label='CTRL-SAC Multi-Traj')

plt.legend(fontsize='xx-large')
plt.savefig("/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/output.png")