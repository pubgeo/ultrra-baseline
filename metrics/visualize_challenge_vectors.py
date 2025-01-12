import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

def plot_threshold_curve(thresholds, scores, labels, title, threshold_label, font_size=32, output_fname=None):
    # define colors and markers
    colors = ['b','g','r','c','m','y','k']
    markers = ['.','^','v','s']
    fmt=[]
    for m in markers:
        for c in colors:
            fmt.append(m+'-'+c)
    # get counts less than threshold
    fractions_list = []
    for ndx in range(len(labels)):
        fractions = []
        for threshold in thresholds:
            metrics = []
            metrics_lists = scores[ndx]
            for metrics_list in metrics_lists:
                metrics.extend(metrics_list)
            metrics = np.array(metrics)
            fractions.append(np.sum(metrics < threshold)/len(metrics))
        fractions_list.append(fractions)
    # plot the curves
    fig, ax = plt.subplots(figsize=(16, 10), layout='constrained')
    for ndx in range(len(labels)):
#        (line,) = ax.plot(thresholds, fractions_list[ndx], "-")
        (line,) = ax.plot(thresholds, fractions_list[ndx], fmt[ndx])
        line.set_label(labels[ndx])
    ax.legend(fontsize=font_size, ncols=2, loc='upper right')
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, max(thresholds))
    ax.grid(True)
    ax.set_xlabel(threshold_label, fontsize=font_size)
    ax.set_ylabel('Fraction Meeting Threshold', fontsize=font_size)
    plt.title(title, fontsize=font_size)
    if output_fname is not None:
        plt.savefig(output_fname)
    else:
        plt.show()

def boxplot(data, xlabel, ylabel, ylim, title, xticks=None, log_plot=False, output_fname=None):
    fig, ax = plt.subplots()
    bp = ax.boxplot(data)
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.2)
    if xticks is not None:
        plt.xticks(np.arange(len(xticks)), xticks)
    plt.xlabel(xlabel)
    if log_plot:
        plt.yscale('log')
    else:
        plt.yscale('linear')
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.title(title)
    if output_fname is not None:
        plt.savefig(output_fname)
    else:
        plt.show()
    plt.close(fig)

def plot_ultrra_results(json_path, output_path):
    # read JSON outputs
    with open(json_path) as f:
        results = json.load(f)
    # plot camera calibration box plots
    print('Producing camera calibration box plots...')
    scores = []
    labels = []
    if 'camera_calibration' in results:
        ylim = [0.01,1000.0]
        keys = list(results['camera_calibration'].keys())
        data = []
        last_step = keys[0][9:11]
        last_vector = keys[0][0:7]
        xticks = ['']
        for key in tqdm(keys):
            # add next step
            new_data = results['camera_calibration'][key]
            if new_data is None:
                new_data = np.array([np.nan])
            else:
                new_data = np.array(new_data)
            new_data[np.isinf(new_data)] = ylim[1]
            step = key[9:11]
            vector = key[0:7]
            # if not in same challenge vector, then plot results and start the next vector
#            if step < last_step:
            if vector != last_vector:
                output_fname = os.path.join(output_path, last_vector + '_camera_calibration_boxplots.png')
                scores.append(data)
                labels.append(last_vector)
                boxplot(data, 'Steps', 'Spherical Error (meters)', ylim, 'Camera Calibration', xticks=xticks, log_plot=True, output_fname=output_fname)
                data = []
                xticks = ['']
            last_step = step
            last_vector = vector
            data.append(new_data)
            xticks.append(step)
        output_fname = os.path.join(output_path, last_vector + '_camera_calibration_boxplots.png')
        scores.append(data)
        labels.append(last_vector)
        boxplot(data, 'Steps', 'Spherical Error (meters)', ylim, 'Camera Calibration', xticks=xticks, log_plot=True, output_fname=output_fname)
    # plot camera calibration threshold curves
    print('Producing camera calibration threshold curves...')
    thresholds = 5.0 * np.array(list(range(101))) / 100.0
    output_fname = os.path.join(output_path, 'camera_calibration_threshold_curves.png')
    plot_threshold_curve(thresholds, scores, labels, 'Camera Calibration', 'Camera Geolocation Error Threshold (m)', output_fname=output_fname)
    # plot view synthesis box plots
    print('Producing view synthesis box plots...')
    scores = []
    labels = []
    if 'view_synthesis' in results:
        ylim = [0,1]
        keys = list(results['view_synthesis'].keys())
        data = []
        last_step = keys[0][9:11]
        last_vector = keys[0][0:7]
        xticks = ['']
        for key in tqdm(keys):
            # add next step
            new_data = np.array(results['view_synthesis'][key])
            step = key[9:11]
            vector = key[0:7]
            # if not in same challenge vector, then plot results and start the next vector
#            if step < last_step:
            if vector != last_vector:
                output_fname = os.path.join(output_path, last_vector + '_view_synthesis_boxplots.png')
                scores.append(data)
                labels.append(last_vector)
                boxplot(data, 'Steps', 'DreamSim Score', ylim, 'View Synthesis', xticks=xticks, output_fname=output_fname)
                data = []
                xticks = ['']
            last_step = step
            last_vector = vector
            data.append(new_data)
            xticks.append(step)
        output_fname = os.path.join(output_path, last_vector + '_view_synthesis_boxplots.png')
        scores.append(data)
        labels.append(last_vector)
        boxplot(data, 'Steps', 'DreamSim Score', ylim, 'View Synthesis', xticks=xticks, output_fname=output_fname)
    # plot view synthesis threshold curves
    print('Producing view synthesis threshold curves...')
    thresholds = np.array(list(range(101))) / 100.0
    output_fname = os.path.join(output_path, 'view_synthesis_threshold_curves.png')
    plot_threshold_curve(thresholds, scores, labels, 'View Synthesis', 'DreamSim Threshold', output_fname=output_fname)

if __name__ == "__main__":
    json_path = sys.argv[1]
    output_path = sys.argv[2]
    plot_ultrra_results(json_path, output_path)

