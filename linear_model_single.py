'''
    Creates and evaluates a linear regression prediction model using a
    single drone (in this example we use DJI Mavic data + accompanying wind data)
'''

from tensorflow.python.keras.metrics import Metric
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from matplotlib import gridspec
import tensorflow as tf
import pandas as pd
import numpy as np
import itertools
import math
import sys
import os

base_loc = os.path.dirname(os.path.realpath(__file__))
data_loc = base_loc + "\\data\\"

sys.path.append(base_loc)
from ml_plotter import *
from oneHz_LSTM import plot_single_moving_avg, split_wspd, find_turb_var, scale, MBE, rmse, findTurbulence



mav_cols  = ['mav_roll', 'mav_pitch']
#mav_cols  = ['roll', 'pitch']
mav_cols = ['mav_roll', 'mav_pitch', 'mav_acc_x', 'mav_acc_y', 'mav_acc_z']
#solo_cols = ['solo_roll', 'solo_pitch']
solo_cols = ['solo_roll', 'solo_pitch', 'solo_acc_x', 'solo_acc_y', 'solo_acc_z']
'''
The coefficient of determination is calculated based on the sum of squared errors divided
 by the total squared variation of y values from their average value. That calculation yields
  the fraction of variation in the dependent variable not captured by the model. 
  Thus the coefficient of variation is 1 — that value. Or, in math terms:

r² = 1 — (Sum of squared errors) / (Total sum of squares)

(Total sum of squares) = Sum(y_i — mean(y))²

(Sum of squared errors) = sum((Actual — Prediction)²)
'''

splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9] # ratio of data used for training
splits = [0.6]
show_figs = 0
save_figs = 1

def main():

    trun = 11 # landing at the end has undesired spikey data
    lf_mav = np.array(pd.read_csv(data_loc + "\\LF-concatenated_mav-sonic.csv")[mav_cols])[:-trun]
    lf_mav_dir = np.array(pd.read_csv(data_loc + "\\LF-concatenated_mav-sonic.csv")['mav_dir'])[:-trun]
    lf_wind = np.array(pd.read_csv(data_loc + '\\LF-truncated_sonic.csv')[['windspeed']])[:-trun]
    lf_wind_dir = np.array(pd.read_csv(data_loc + '\\LF-truncated_sonic.csv')[['wind_direction']])[:-trun]

    # drone_attitude = np.array(pd.read_csv(data_loc + "\\032620-concat-mav.csv")[mav_cols])
    # ground_truth_wspd = np.array(pd.read_csv(data_loc + "\\032620-concat-mav.csv")['sonic_wind'])
    # ground_truth_wdir = np.array(pd.read_csv(data_loc + "\\032620-concat-mav.csv")['sonic_wind_dir'])

    # soloAttToUse,mavicAttToUse = lf_solo, lf_mav
    # soloWindToUse,mavicWindToUse = lf_wind, lf_wind
    #soloDirToUse, mavicDirToUse = lf_solo_dir, lf_mav_dir

    sonicDirToUse = lf_wind_dir
    selected_mav = lf_mav
    selected_mav_wind = lf_wind
    mavicDirToUse = lf_mav_dir
    #selected_solo = lf_solo
    #selected_solo_wind = lf_wind


    #for i in range(len(solo_cols)):
    #    selected_solo[:,i]  = scale(selected_solo[:,i])

    for i in range(len(mav_cols)):
        selected_mav[:,i] = scale(selected_mav[:,i])
        
    #wind_scaled    = scaler.fit_transform(wind)

    trial_metrics = []
    # Split Test and Train Set 
    for split in splits:
        mavic_training_split = int(len(selected_mav) * split)
        #solo_training_split  = int(len(selected_solo) * split)

        mavic_wind_dir = mavicDirToUse[mavic_training_split:]
        #solo_wind_dir = soloDirToUse[solo_training_split:]
        validation_wind_dir = sonicDirToUse[mavic_training_split:]

        # Split the prepared datasets into train sets and test/validation sets 
        X_train_mavic, X_test_mavic = selected_mav[:mavic_training_split], selected_mav[mavic_training_split:]
        y_train_mavic, y_test_mavic = selected_mav_wind[:mavic_training_split], selected_mav_wind[mavic_training_split:]

        #X_train_solo, X_test_solo = selected_solo[:solo_training_split], selected_solo[solo_training_split:]
        #y_train_solo, y_test_solo = selected_solo_wind[:solo_training_split], selected_solo_wind[solo_training_split:]

        mav_model  = LinearRegression().fit(X_train_mavic, y_train_mavic)
        #solo_model = LinearRegression().fit(X_train_solo, y_train_solo)

        mav_sq = mav_model.score(X_test_mavic, y_test_mavic)
        #solo_sq = solo_model.score(X_test_solo, y_test_solo)
        #print(' mav / solo coefficient of determination: %.03f/%.03f' % (mav_sq, solo_sq))
        print('mav coefficient of determination: %.03f' % (mav_sq))

        mav_pred = mav_model.predict(X_test_mavic)
        #solo_pred = solo_model.predict(X_test_solo)
        #metrics = plot_wiggles("solo linear march", y_train_solo, y_test_solo, solo_pred,split)
        metrics = plot_wiggles("mavic linear march", y_train_mavic, y_test_mavic, mav_pred, split)
        trial_metrics.append(metrics)

        pred_u, pred_v = split_wspd(mav_pred, mavic_wind_dir)
        actual_u, actual_v = split_wspd(y_test_mavic, validation_wind_dir)
        p_u, p_v = find_turb_var(pred_u, pred_v)
        a_u, a_v = find_turb_var(actual_u, actual_v)
        print("-- mavic --\nu turb diff: %.03f\nv turb diff: %.03f\n" % (a_u - p_u, a_v - p_v))

        # pred_u, pred_v = split_wspd(solo_pred, solo_wind_dir)
        # actual_u, actual_v = split_wspd(y_test_solo, validation_wind_dir)
        # p_u, p_v = find_turb_var(pred_u, pred_v)
        # a_u, a_v = find_turb_var(actual_u, actual_v)
        # print("-- solo --\nu turb diff: %.03f\nv turb diff: %.03f\n" % (a_u - p_u, a_v - p_v))

        drone_pre = ""
        #transposedTest = [list(ll) for ll in zip(*testData)]
        drone_type = 1
        for drone in range(2):
            transposedColumns = []
            drone_cols = []
            if drone_type == -1:
                pass
                # drone_cols = solo_cols
                # drone_pre = "solo"
                # testData = X_test_solo
                # drone_wspd_pred = solo_pred
                # drone_dir = solo_wind_dir
                # actual_wspd = y_test_solo
            else:
                drone_cols = mav_cols
                drone_pre = "mav"
                testData = X_test_mavic
                drone_wspd_pred = mav_pred
                drone_dir = mavic_wind_dir
                actual_wspd = y_test_mavic

            for d, dc in enumerate(drone_cols): #0-5
                transposedCol = []
                for dp in testData:#0-851
                    transposedCol.append(dp[d])
                transposedColumns.append(transposedCol)

            label_df = pd.DataFrame(data=[], columns=drone_cols)
            pred_cols = ['%s_predicted_wspd' % drone_pre, '%s_est_wdir' % drone_pre, 'sonic_wspd', 'sonic_wdir']
            prediction_df = pd.DataFrame(data=[], columns= drone_cols + pred_cols)
            pred_data = (drone_wspd_pred, drone_dir, actual_wspd, validation_wind_dir)

            # store prediction data 
            for tc in range(len(transposedColumns)):
                col_data = list(transposedColumns[tc][:])
                col_label = drone_cols[tc]
                col_df = pd.DataFrame({col_label:col_data})
                prediction_df[col_label] = col_data

            for pc in range(len(pred_cols)):
                col_label = pred_cols[pc] 
                try:
                    prediction_df[col_label] = pred_data[pc]
                except:
                    prediction_df[col_label] = pred_data[pc][0]


            prediction_df.to_csv(base_loc + "%s-march-linear-validation.csv" % drone_pre, sep=",", index=False)
            drone_type *= -1

        title = "Linear March Comparison"
        labels = ["Mavic Wspd" , "True Wspd"]
        y_lims = [0, 11]
        # try:
        #     plot_mult_moving_avg(solo_pred, mav_pred, y_test_mavic, labels, "Windspeed (m s$^{-1}$)", y_lims, title, show_metrics=1)
        # except:
        #     plot_mult_moving_avg(list(itertools.chain.from_iterable(solo_pred)), list(itertools.chain.from_iterable(mav_pred)), list(itertools.chain.from_iterable(y_test_mavic)), labels, "Windspeed (m s$^{-1}$)", y_lims, title, show_metrics=1)

        try:
            plot_single_moving_avg(preds[0], preds[1], list(itertools.chain.from_iterable(actual_wspd)), labels, "Windspeed (m s$^{-1}$)", y_lims, title, show_metrics=1)
        except:
            plot_single_moving_avg(list(itertools.chain.from_iterable(mav_pred)), list(itertools.chain.from_iterable(y_test_mavic)), labels, "Windspeed (m s$^{-1}$)", y_lims, title, show_metrics=1)
            

def plot_wiggles(title,train,test,pred, ti):
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 2])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    axes = [ax0, ax1]
    a = plt.gca()
    a.set_ylim([0, 9])  

    time = [i for i in range(len(train) + len(test))]
    axes[0].plot(time[:len(train)], train, label='History', color='#00CAF0')
    axes[0].plot(time[len(time)-len(test):], test, label='True Wspd', color='#E57200',linewidth=1.2)
    axes[0].plot(time[len(time)-len(pred):], pred, label='Predicted Wspd', color='#232D4B',alpha=0.8,linewidth=1.2)
    axes[1].plot(time[len(time)-len(test):], test, label='True Wspd', color='#E57200', linewidth=1.6)
    axes[1].plot(time[len(time)-len(pred):], pred, label='Predicted Wspd', color='#232D4B',alpha=0.8, linewidth=1.6)
    fig.tight_layout()
    axes[0].set_title(title)

    plt.ylabel("wind speed (m/s)", fontsize=19)
    a.yaxis.set_label_coords(-0.05,1.0)
    ax0.set_xlabel("Time (s)", fontsize=19)
    ax1.set_xlabel("Time (s)", fontsize=19)

    #ax0.get_shared_x_axes().join(ax0, ax1)
    #ax0.set_xticklabels([])

    ax0.legend(loc="best", ncol=3)
    ax1.legend(loc="best", ncol=2)
    plt.subplots_adjust(top=0.92,bottom=0.14,left=0.1,right=0.97,hspace=0.3,wspace=0.06)

    print("-----------linear-------------\n%s\n" % title)
    print('{:34s}: {:4.2f}'.format("true wind speed variance", np.std(test)**2))
    print('{:34s}: {:4.2f}'.format("predicted wind speed variance", np.std(pred)**2))
    print('{:34s}: {:4.2f}'.format("wind speed variance difference", np.std(test)**2 - np.std(pred)**2))
    print('{:34s}: {:4.2f}'.format("MBE",MBE(test,pred)))
    print('{:34s}: {:4.2f}'.format("MSE",mean_squared_error(test,pred)))
    print('{:34s}: {:4.2f}'.format("RMSE",np.sqrt(mean_squared_error(test,pred))))

    print('{:34s}: {:4.2f}'.format("true turbulence intensity",findTurbulence(test)))
    print('{:34s}: {:4.2f}'.format("predicted turbulence intensity",findTurbulence(pred)))
    print('{:34s}: {:4.2f}'.format("turbulence intensity differenece",findTurbulence(test)-findTurbulence(pred)))

    if show_figs:
        plt.show()
    plt.close()

    try:
        metrics = plot_moving_avg_final(list(itertools.chain.from_iterable(test)), list(itertools.chain.from_iterable(pred)), title, ti)
    except:
        metrics = plot_moving_avg_final(test, pred, title, ti)
    return metrics
    

if __name__ == "__main__":
    main()
