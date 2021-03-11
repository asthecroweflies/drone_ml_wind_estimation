'''

    Utility script for plotting time series of predictions vs. expectations
    in addition to history of data. 

'''

from tensorflow.python.keras.metrics import Metric
from sklearn.metrics import mean_squared_error
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from matplotlib import gridspec
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import itertools
import time 
import os

base_loc = os.path.dirname(os.path.realpath(__file__))
data_loc = base_loc + "\\data\\"

show_figs = 1
save_figs = 0

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams['font.size']   = 12

def plot_moving_avg_final(t1, t2, t, ti):
    window_length = 10
    try:
        t1_avg = np.convolve(t1, np.ones((window_length,))/window_length, mode='valid')
    except:
        t1_avg = np.convolve(list(itertools.chain.from_iterable(t1)), np.ones((window_length,))/window_length, mode='valid')
    try:
        t2_avg = np.convolve(t2, np.ones((window_length,))/window_length, mode='valid')
    except:
        t2_avg = np.convolve(list(itertools.chain.from_iterable(t2)), np.ones((window_length,))/window_length, mode='valid')
    
    fig = plt.figure(figsize=(8.75, 3.25)) 
    gs = gridspec.GridSpec(1, 1, height_ratios=[1]) 
    ax0 = plt.subplot(gs[0])

    axes = [ax0]#, ax1]
    a = plt.gca()
    ylim = [0, max(t1) * 1.3]
    axes[0].set_ylim(ylim)
    t1_min = np.divide(range(len(t1_avg)), 60)
    t2_min = np.divide(range(len(t2_avg)), 60)

    axes[0].plot(t1_min, t1_avg, label='True Wspd', color='#E57200',alpha=0.9,linewidth=2.42)
    axes[0].plot(t2_min, t2_avg, label='Predicted Wspd', color='#232D4B',alpha=0.8,linewidth=2.42)

    fig.tight_layout()

    metrics = [MBE(t1_avg,t2_avg), np.sqrt(mean_squared_error(t1_avg,t2_avg))]

    print("------------------------\n%s\n" % "moving avg")
    print('{:34s}: {:4.2f}'.format("true wind speed variance", np.std(t1_avg)**2))
    print('{:34s}: {:4.2f}'.format("predicted wind speed variance", np.std(t2_avg)**2))
    print('{:8s}: {:4.3f}'.format("MBE",MBE(t1_avg,t2_avg)))
    print('{:8s}: {:4.3f}'.format("RMSE",np.sqrt(mean_squared_error(t1_avg,t2_avg))))

    # Set labels
    metric_label = "RMSE: %.02f m s$^{-1}$ MBE: %.02f m s$^{-1}$" % (metrics[1], metrics[0])
    axes[0].annotate(metric_label, 
                    (0.02, 0.975),
                    xytext=(4, -4),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    fontweight='bold',
                    color='k',
                    backgroundcolor='white',
                    ha='left', va='top')

    axes[0].set_ylabel("Windspeed (m s$^{-1}$)", fontsize=12)
    ax0.set_xlabel("Time (minutes)", fontsize=12)
    axes[0].legend(loc="best", ncol=2,frameon=False)
    plt.subplots_adjust(top=0.885,bottom=0.194,left=0.1,right=0.94,hspace=0.2,wspace=0.06)
    if save_figs:
        plt.savefig(base_loc + "\\%s_%d.png" % (t, ti * 100))

    if show_figs:
        plt.show()
    plt.close()
    return metrics

def plot_metrics_final(trial_metrics, training_increments, dataLength, title):
    metric_labels = ["MBE", "RMSE"]
    markers = ["-o","-s"]
    colors  = ['#E57200','#373F95']
    #fig = plt.figure(figsize=(10,7.5))
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Length of Training Data (m)")
    ax2 = ax1.twinx()
    trans = np.transpose(trial_metrics)
    time_training_axis = np.divide(np.multiply(training_increments, dataLength), 60)

    mbe = trans[0]
    ax2.plot(time_training_axis, mbe, markers[0], color=colors[0],
            markersize=9, linewidth=2.1,
            markerfacecolor='white',
            markeredgecolor=colors[0],
            markeredgewidth=2, label=metric_labels[0])

    ax2.set_ylabel('MBE')
    ax2.set_ylim([-1,1])
    rmse = trans[1]
    ax1.plot(time_training_axis, rmse, markers[1], color=colors[1],
            markersize=9, linewidth=2.1,
            markerfacecolor='white',
            markeredgecolor=colors[1],
            markeredgewidth=2, label=metric_labels[1])

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    print("\n--- metrics ---")
    print(trans)

    ax1.set_ylabel('RMSE (m $s^{-1}$)')
    ax1.set_ylim([0,1])

    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax1.legend(lines, labels, ncol=2, loc="upper right")
    #plt.legend(loc="upper right", ncol=2)

    plt.title(title)
    #if show_figs:
    plt.subplots_adjust(top=0.88,bottom=0.11,left=0.135,right=0.88,hspace=0.2,wspace=0.2)

    plt.grid(True)
    print("%s\nRMSE:" % title)
    print(rmse)
    print("MBE:\n")
    print(mbe)

    plt.show()

    if save_figs:
        plt.savefig(base_loc + "%s_spread.png" % (title))

def MBE(y_true, y_pred):
    '''
    Parameters:
        y_true (array): Array of observed values
        y_pred (array): Array of prediction values

    Returns:
        mbe (float): bias score
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.reshape(len(y_true),1)
    y_pred = y_pred.reshape(len(y_pred),1)   
    diff = (y_pred - y_true)
    mbe = diff.mean()
    return round(mbe,3)