import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description="Run the analysis script with customizable parameters.")
# Add arguments
parser.add_argument("--experiment", type=str, default="OOD", choices=["legeval", "legood", "OOD", "legtrain"], help="Specify the task name (default: OOD).")
args = parser.parse_args()

output_dir = f"runs/experiments/{args.experiment}"

agents = [entry.name for entry in os.scandir(output_dir) if entry.is_dir()]

for agent in agents:
    print(agent)