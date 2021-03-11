'''
    Given data for two drones and accompanying windspeeds,
    this script produces a Sklearn's KNN model used for predicting windspeeds
    for a given set of attitude data.

'''

import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from tqdm import trange
from timeit import default_timer as timer
from sklearn.model_selection import KFold
import matplotlib.pylab as pylab
import itertools
import sys
import os

base_loc = os.path.dirname(os.path.realpath(__file__))
data_loc = base_loc + "\\data\\"
#mav_cols  = ['mav_roll', 'mav_pitch']
mav_cols = ['mav_roll', 'mav_pitch', 'mav_acc_x', 'mav_acc_y', 'mav_acc_z']
#solo_cols = ['solo_roll', 'solo_pitch']
solo_cols = ['solo_roll', 'solo_pitch', 'solo_acc_x', 'solo_acc_y', 'solo_acc_z']

splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9]
splits = [0.6]

show_figs = 0
save_figs = 1

sys.path.append(base_loc)
from oneHz_LSTM import plot_mult_moving_avg, split_wspd, find_turb_var, scale, MBE, rmse, findTurbulence

def main():
    params = {'legend.fontsize': 'large',
            'axes.labelsize': 'x-large',
            'xtick.labelsize':'x-large',
            'ytick.labelsize':'x-large',
            'font.family':'sans-serif'}
    pylab.rcParams.update(params)

    trun = 11
    lf_mav = np.array(pd.read_csv(data_loc + "\\LF-concatenated_mav-sonic.csv")[mav_cols])[:-trun]
    lf_solo = np.array(pd.read_csv(data_loc + "\\LF-concatenated_solo-sonic.csv")[solo_cols])[:-trun]
    lf_mav_dir = np.array(pd.read_csv(data_loc + "\\LF-concatenated_mav-sonic.csv")['mav_dir'])[:-trun]
    lf_solo_dir = np.array(pd.read_csv(data_loc + "\\LF-concatenated_solo-sonic.csv")['solo_dir'])[:-trun]
    lf_wind = np.array(pd.read_csv(data_loc + '\\LF-truncated_sonic.csv')[['windspeed']])[:-trun]
    lf_wind_dir = np.array(pd.read_csv(data_loc + '\\LF-truncated_sonic.csv')[['wind_direction']])[:-trun]

    soloAttToUse,mavicAttToUse = lf_solo, lf_mav
    soloWindToUse,mavicWindToUse = lf_wind, lf_wind
    soloDirToUse, mavicDirToUse = lf_solo_dir, lf_mav_dir
    sonicDirToUse = lf_wind_dir
    
    soloAttToUse,mavicAttToUse = lf_solo, lf_mav
    soloWindToUse,mavicWindToUse = lf_wind, lf_wind

    for i in range(len(solo_cols)):
        soloAttToUse[:,i]  = scale(soloAttToUse[:,i])

    for i in range(len(mav_cols)):
        mavicAttToUse[:,i] = scale(mavicAttToUse[:,i])

    mavic_scaled = mavicAttToUse
    solo_scaled = soloAttToUse
    trial_metrics = []
    drone_type = -1

    for s in splits:
        # Split Test and Train Set 
        mavic_training_split = int(len(mavic_scaled) * s)
        solo_training_split  = int(len(solo_scaled) * s)

        mavic_wind_dir = mavicDirToUse[mavic_training_split:]
        solo_wind_dir = soloDirToUse[solo_training_split:]
        validation_wind_dir = sonicDirToUse[mavic_training_split:]

        X_train_mavic_knn, X_test_mavic_knn = mavic_scaled[:mavic_training_split], mavic_scaled[mavic_training_split:]
        y_train_mavic_knn, y_test_mavic_knn = np.array(mavicWindToUse)[:mavic_training_split], np.array(mavicWindToUse)[mavic_training_split:]
        X_train_solo_knn, X_test_solo_knn = solo_scaled[:solo_training_split], solo_scaled[solo_training_split:]
        y_train_solo_knn, y_test_solo_knn = soloWindToUse[:solo_training_split], np.array(soloWindToUse)[solo_training_split:]
        print("len training: %.02f len vali: %.02f" % (len(X_train_solo_knn), len(X_test_solo_knn)))

        rmse_val = []
        optimal_solo_k_rmse, optimal_mavic_k_rmse = -1, -1
        optimal_solo_k_mbe, optimal_mavic_k_mbe = -1, -1
        mav_mbe, solo_mbe = 1e10, 1e10
        mav_best_mbe = 1e9
        solo_best_mbe = 1e9
        mav_rmse = 1e9
        solo_rmse = 1e9
        mav_best_rmse, solo_best_rmse = 1e10, 1e10
        solo_mbes, mav_mbes, solo_rmses, mav_rmses = [],[],[],[]
        
        for K in trange(1, 100):
            mav_model = neighbors.KNeighborsRegressor(n_neighbors = K)
            solo_model = neighbors.KNeighborsRegressor(n_neighbors = K)

            solo_model.fit(X_train_solo_knn, y_train_solo_knn)  #fit the model
            mav_model.fit(X_train_mavic_knn, y_train_mavic_knn)  #fit the model

            solo_pred = solo_model.predict(X_test_solo_knn) #make prediction on test set
            mav_pred = mav_model.predict(X_test_mavic_knn) #make prediction on test set

            solo_rmse = mean_squared_error(y_test_solo_knn, solo_pred) #calculate rmse
            mav_rmse = mean_squared_error(y_test_mavic_knn, mav_pred) 
            solo_mbe = MBE(y_test_solo_knn, solo_pred)
            mav_mbe = MBE(y_test_mavic_knn, mav_pred)
            solo_mbes.append(solo_mbe)
            solo_rmses.append(solo_rmse)
            mav_mbes.append(mav_mbe)
            mav_rmses.append(mav_rmse)

            if abs(solo_rmse) < abs(solo_best_rmse):
                solo_best_rmse = solo_rmse
                optimal_solo_k_rmse = K
            if abs(mav_rmse) < abs(mav_best_rmse):
                mav_best_rmse = mav_rmse
                optimal_mavic_k_rmse = K
            if abs(solo_mbe) < abs(solo_best_mbe):
                solo_best_mbe = solo_mbe
                optimal_solo_k_mbe = K
            if abs(mav_mbe) < abs(mav_best_mbe):
                mav_best_mbe = mav_mbe
                optimal_mavic_k_mbe = K
        print('optimal solo rmse K: %d, %.02f\noptimal solo mbe K: %d, %.02f\noptimal mav rmse K: %d, %.02f\noptimal mav mbe K: (%d, %.02f)' %\
            (optimal_solo_k_rmse, solo_best_rmse, optimal_solo_k_mbe, solo_best_mbe, optimal_mavic_k_rmse, mav_best_rmse, optimal_mavic_k_mbe, mav_best_mbe))


        start_time = timer()

        mavic_knn = neighbors.KNeighborsRegressor(n_neighbors = optimal_mavic_k_rmse)
        mavic_knn.fit(X_train_mavic_knn, y_train_mavic_knn)
        solo_knn = neighbors.KNeighborsRegressor(n_neighbors = optimal_solo_k_rmse)
        solo_knn.fit(X_train_solo_knn, y_train_solo_knn)  

        mav_pred = mavic_knn.predict(X_test_mavic_knn)
        solo_pred = solo_knn.predict(X_test_solo_knn)
        end_time = timer()

        title = "KNN March Comparison"
        labels = ["Solo Wspd", "Mavic Wspd" , "True Wspd"]
        y_lims = [0, 12]
        try:
            plot_mult_moving_avg(solo_pred, mav_pred, y_test_mavic_knn, labels, "Windspeed (m s$^{-1}$)", y_lims, title, show_metrics=1)
        except:
            plot_mult_moving_avg(list(itertools.chain.from_iterable(solo_pred)), list(itertools.chain.from_iterable(mav_pred)), list(itertools.chain.from_iterable(y_test_mavic_knn)), labels, "Windspeed (m s$^{-1}$)", y_lims, title, show_metrics=1)

    plt.clf()
    plt.cla()
    plt.close()

if __name__ == "__main__":
    main()
