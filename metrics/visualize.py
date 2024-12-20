import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def boxplot(data, xlabel, ylabel, ylim, title, xticks=None, log_plot=False, output_fname=None):
    fig, ax = plt.subplots()
    bp = ax.boxplot(data)
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.2)
    if xticks is not None:
        plt.xticks(np.arange(len(xticks)), xticks)
    else:
        plt.xlabel(xlabel)
    if log_plot:
        plt.yscale('log')
    else:
        plt.yscale('linear')
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.title(title)
    if output_path is not None:
        plt.savefig(output_fname)
    else:
        plt.show()

def plot_ultrra_results(json_path, output_path):
    # read JSON outputs
    with open(json_path) as f:
        results = json.load(f)
    # plot camera calibration results
    if 'camera_calibration' in results:
        output_fname = os.path.join(output_path, 'camera_calibration_results.png')
        keys = list(results['camera_calibration'].keys())
        print(keys)
        data = []
        for key in keys:
            new_data = np.array(results['camera_calibration'][key])
            data.append(new_data)
        # swap themes 3 and 4
        data[2], data[3] = data[3], data[2]
        xticks = ['', 'Single\nCamera', 'Multiple\nCameras', 'Varying\nAltitudes', 'Reconstructed\nArea']
        ylim = [0.01,1000.0]
        boxplot(data, 'Themes', 'Spherical Error (meters)', ylim, 'Camera Calibration', xticks=xticks, log_plot=True, output_fname=output_fname)
    # plot view synthesis results
    if 'view_synthesis' in results:
        output_fname = os.path.join(output_path, 'view_synthesis_results.png')
        keys = list(results['view_synthesis'].keys())
        print(keys)
        data = []
        for key in keys:
            new_data = np.array(results['view_synthesis'][key])
            data.append(new_data)
        # swap themes 3 and 4
        data[2], data[3] = data[3], data[2]
        xticks = ['', 'Single\nCamera', 'Multiple\nCameras', 'Varying\nAltitudes', 'Reconstructed\nArea']
        ylim = [0,1]
        boxplot(data, 'Themes', 'DreamSim Score', ylim, 'View Synthesis', xticks=xticks, output_fname=output_fname)

def plot_test():
    data1 = np.random.normal(100, 10, 200)/600.
    data2 = np.random.normal(80, 30, 200)/300.
    data3 = np.random.normal(90, 20, 200)/200.
    data4 = np.random.normal(90, 5, 200)/100.
    data = [data1, data2, data3, data4]
    xticks = ['', 'TO1', 'T02', 'TO3', 'TO4']
    ylim = [0,1]
    boxplot(data, 'Themes', 'DreamSim Scores', ylim, 'Development Results', xticks)

# test
if __name__ == "__main__":
    json_path = sys.argv[1]
    output_path = sys.argv[2]
    plot_ultrra_results(json_path, output_path)

