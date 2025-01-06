import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
tasks = ["OOD"]
for i, task in enumerate(tasks):
    folder = f"/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/{task}/"
    print(folder)
    ref_traj = np.loadtxt(f"{folder}trajectory.txt",
                 delimiter=" ", dtype=float)

    sac_agent = np.loadtxt(f"{folder}sac_agent_positions.txt",
                 delimiter=" ", dtype=float)
    
    # ctrl_agent = np.loadtxt(f"{folder}ctrl_True_agent_positions.txt",
    #             delimiter=" ", dtype=float)
    
    print(ref_traj.shape)
    print(sac_agent.shape)
    # print(ctrl_agent.shape)
    
    axs.scatter(ref_traj[:300, 0], ref_traj[:300, 1], s= 15, alpha=1, label='Reference Trajectories')
    axs.scatter(sac_agent[:, 0], sac_agent[:, 1], marker='x',alpha=0.3, s=10, label='SAC Multi-Traj')
    # axs[i].scatter(ctrl_agent[:500, 0], ctrl_agent[:500, 1], marker='x',alpha=0.3, s=40, label='CTRL-SAC Multi-Traj')

plt.legend(fontsize='xx-large')
plt.savefig("/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/output.png")