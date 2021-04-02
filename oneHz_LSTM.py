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
import sys
import os

sys.path.append("G:\\My Drive\\wind_est-ml\\data manip\\")
from ml_plotter import plot_moving_avg_final
from ml_plotter import plot_metrics_final

epochs = 50     # duration of training
show_figs = 1   
save_figs = 0
training_increments = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # determines training / validation split
training_increments = [0.6] # we found comparable results using ratios down to ~40:60
use_moving_avg_metrics = 1
use_all_cols = 1            # whether to include accelerometer data or only roll + pitch 
solo_cols, mav_cols = [], []
predicted_wspd, actual_wspd = [], []

base_loc = os.path.dirname(os.path.realpath(__file__))
data_loc = base_loc + "\\data\\"

if use_all_cols:
    solo_cols = ['solo_roll', 'solo_pitch', 'solo_acc_x', 'solo_acc_y', 'solo_acc_z']
    mav_cols  = ['mav_roll', 'mav_pitch', 'mav_acc_x', 'mav_acc_y', 'mav_acc_z']

else:
    solo_cols = ['solo_roll', 'solo_pitch']
    mav_cols  = ['mav_roll', 'mav_pitch']

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams['font.size']   = 12

def main():

    '''
    Example data for 1 drone with corresponding wind data provided
    '''

    drone_attitude = np.array(pd.read_csv(data_loc + "\\032620-concat-mav.csv")[mav_cols])
    ground_truth_wspd = np.array(pd.read_csv(data_loc + "\\032620-concat-mav.csv")['sonic_wind'])
    ground_truth_wdir = np.array(pd.read_csv(data_loc + "\\032620-concat-mav.csv")['sonic_wind_dir'])

    title = "1 Hz March Mavic"

    for m in range(len(mav_cols)):
        drone_attitude[:,m] = scale(drone_attitude[:,m])

    trials = [(drone_attitude, ground_truth_wspd,  title, ground_truth_wdir)]
    
    '''
    For each tuple in trials, trains a model on given data and plots predictions on test data
    '''
    models = []
    trial_metrics = []
    trial_results = [] # contains predictions and ground truths 
    best_training_increment = 0
    best_rmse = 1e4
    preds = []
    prediction_df = pd.DataFrame() 
    drone_type = 1
    labels_found = 0
    actual_wspd, validation_wind_dir = [], []
    has_dir = 1

    while(trials):
        if has_dir:
            Data, Labels, Title, drone_dir = trials.pop(0)
        else:
            Data, Labels, Title = trials.pop(0)
        Labels = Labels[~np.isnan(Labels)]
        trial_metrics = []
        for ti in training_increments:
            train_split = int(10 * ti)
            idx = train_split*len(Data)//10
            trainData, trainLabels = np.expand_dims(Data[:idx], 1), Labels[:idx]
            testData, testLabels = np.expand_dims(Data[idx:], 1), Labels[idx:]

            print("length of training: " + str(len(Labels) * ti))
            print("length of validation: " +  str(len(Labels) - len(Labels) * ti))
            print("training) wind min: %.03f wind max: %.03f wind mean: %.02f" % (min(Labels[idx:]), max(Labels[idx:]), np.mean(Labels[idx:])))
            print("validation) wind min: %.03f wind max: %.03f wind mean: %.02f" % (min(Labels[:idx]), max(Labels[:idx]), np.mean(Labels[:idx])))

            try:
                drone_dir = drone_dir[idx:]
                validation_wind_dir = ground_truth_wdir[idx:]
            except:
                pass

            model = buildLSTM(Data.shape[1])
            models.append((model, Title))
            #model.save('..\\models\\%s-(%02f)' % (Title, (train_split * 10)))

            if testLabels.shape[0] != testData.shape[0]:
                testLabels = np.pad(testLabels, (0, testData.shape[0] - testLabels.shape[0]), 'constant',\
                                    constant_values=testLabels[len(testLabels)-1][0])

            train(model, trainData, trainLabels, testData, testLabels, Title)
            metrics = plot_predictions(model, testData, trainLabels, testLabels, Title, ti, save_fig=0)

            if (metrics[1] < best_rmse):
                best_rmse = metrics[1]
                best_training_increment = ti

            trial_metrics.append(metrics)
            drone_wspd_pred = model.predict(testData, steps=1)[:,0]
            preds.append(drone_wspd_pred)
            trial_results.append((drone_wspd_pred, Title))

            if has_dir:
                if labels_found == 0:
                    actual_wspd = testLabels
                    labels_found = 1
                pred_u, pred_v = split_wspd(drone_wspd_pred, drone_dir)
                try:
                    actual_u, actual_v = split_wspd(actual_wspd, validation_wind_dir)
                except:
                    actual_u, actual_v = split_wspd(list(itertools.chain.from_iterable(actual_wspd)), validation_wind_dir)

                print("actual turb var: ")
                a_u, a_v = find_turb_var(actual_u, actual_v)

                print("predicted turb var: ")
                p_u, p_v = find_turb_var(pred_u, pred_v)

                print("u turb diff: %.03f\nv turb diff: %.03f" % (a_u - p_u, a_v - p_v))

        if has_dir:
            drone_pre = ""
            #transposedTest = [list(ll) for ll in zip(*testData)]
            transposedColumns = []
            drone_cols = []

            if drone_type == -1:
                drone_cols = solo_cols
                drone_pre = "solo"
            else:
                drone_cols = mav_cols
                drone_pre = "mav"

            for d, dc in enumerate(drone_cols): #0-5
                transposedCol = []
                for dp in testData:#0-851
                    transposedCol.append(dp[0][d])
                transposedColumns.append(transposedCol)

            label_df = pd.DataFrame(data=[], columns=drone_cols)
            pred_cols = ['%s_predicted_wspd' % drone_pre, '%s_est_wdir' % drone_pre, 'sonic_wspd', 'sonic_wdir']
            prediction_df = pd.DataFrame(data=[], columns= drone_cols + pred_cols)
            pred_data = (drone_wspd_pred, drone_dir, actual_wspd, validation_wind_dir)

            # store prediction data 
            for tc in range(len(transposedColumns)):
                col_data     = list(transposedColumns[tc][:])
                col_label    = drone_cols[tc]
                col_df       = pd.DataFrame({col_label:col_data})
                label_df[col_label]      = col_data
                prediction_df[col_label] = col_data

            for pc in range(len(pred_cols)):
                col_label = pred_cols[pc]
                prediction_df[col_label] = pred_data[pc]

            try:
                pred_df = pd.DataFrame({'%s_predicted_wspd' % drone_pre: drone_wspd_pred, '%s_est_wdir' % drone_pre: drone_dir, 'sonic_wspd':actual_wspd, 'sonic_wdir':validation_wind_dir})
            except:
                pred_df = pd.DataFrame({'%s_predicted_wspd' % drone_pre: drone_wspd_pred, '%s_est_wdir' % drone_pre: drone_dir, 'sonic_wspd':list(itertools.chain.from_iterable(actual_wspd)), 'sonic_wdir':list(itertools.chain.from_iterable(validation_wind_dir))})

            label_df = label_df.append(pred_df)
            prediction_df = prediction_df.append(pred_df)
            # Save 
            prediction_df.to_csv(os.getcwd() + "%s-march-lstm-validation.csv" % drone_pre, sep=",", index=False)

        drone_type *= -1

    title = "LSTM march comp"
    labels = ["Mavic Wspd" , "True Wspd"]
    y_lims = [0, 11]

    try:
        plot_single_moving_avg(preds[0], list(itertools.chain.from_iterable(actual_wspd)), labels, "Windspeed (m s$^{-1}$)", y_lims, title, show_metrics=1)
    except:
        plot_single_moving_avg(preds[0], actual_wspd, labels, "Windspeed (m s$^{-1}$)", y_lims, title, show_metrics=1)

    plt.clf()
    plt.cla()
    plt.close()
    dataLength = len(trainLabels) + len(testLabels)
    plot_metrics_final(trial_metrics,training_increments, dataLength, "%s metric spread " % Title)


def plot_single_moving_avg(t1, ground_truth, labels, y_label, y_lims, title, show_metrics):

    window_length = 10

    t1_avg = np.convolve(t1, np.ones((window_length,))/window_length, mode='valid')
    gt_avg = np.convolve(ground_truth, np.ones((window_length,))/window_length, mode='valid')
    lf = min(len(t1_avg), len(gt_avg))
    t1_avg = t1_avg[:lf]
    gt_avg = gt_avg[:lf]

    fig = plt.figure(figsize=(8.75, 3.25)) 
    gs = gridspec.GridSpec(1, 1, height_ratios=[1]) 
    ax0 = plt.subplot(gs[0])
    fig.tight_layout()
    axes = [ax0]#, ax1]
    a = plt.gca()
    axes[0].set_ylim(ymin = 0, ymax = 6)
    axes[0].set_xlim([-0.25,16])

    t1_min = np.divide(range(len(t1_avg)), 60)
    gt_min = np.divide(range(len(gt_avg)), 60)

    colors = ['#232D4B', '#E57200']

    axes[0].plot(gt_min, gt_avg, label=labels[1], color=colors[1],alpha=0.75,linewidth=1.82)
    axes[0].plot(t1_min, t1_avg, label=labels[0], color=colors[0],alpha=0.8,linewidth=1.82)

    # Set labels
    drone_metrics = [MBE(gt_avg,t1_avg), np.sqrt(mean_squared_error(gt_avg,t1_avg))]


    if show_metrics:
        metric_label = ('Drone RMSE:   %.02f m s$^{-1}$  Drone MBE:   %.02f m s$^{-1}$' % (drone_metrics[1], drone_metrics[0]))
        axes[0].annotate(metric_label, 
        (0.01, 0.965),
        xytext=(4, -4),
        xycoords='axes fraction',
        textcoords='offset points',
        fontweight='bold',
        color='k',
        backgroundcolor='white',
        ha='left', va='top')

    #axes[0].set_title("%d training split - %s" % (ti * 100, t))
    axes[0].set_ylabel(y_label, fontsize=18)
    #a.yaxis.set_label_coords(-0.05,-0.1)
    ax0.set_xlabel("Time (minutes)", fontsize=18)
    axes[0].legend(loc="upper right", ncol=1,frameon=True, prop={'size': 13})
    ax0.yaxis.set_ticks(np.arange(0, y_lims[1], 1))
    ax0.tick_params(axis='both', which='major', labelsize=12)
    plt.subplots_adjust(top=0.885,bottom=0.194,left=0.1,right=0.94,hspace=0.2,wspace=0.06)

    if save_figs:
        plt.savefig(base_loc +"\\single_trial_%s.png" % (title))

    plt.grid(linestyle="--")
    if show_figs:
        plt.show()
    
    plt.close()

def plot_mult_moving_avg(t1, t2, ground_truth, labels, y_label, y_lims, title, show_metrics):
    window_length = 10

    t1_avg = np.convolve(t1, np.ones((window_length,))/window_length, mode='valid')
    t2_avg = np.convolve(t2, np.ones((window_length,))/window_length, mode='valid')
    gt_avg = np.convolve(ground_truth, np.ones((window_length,))/window_length, mode='valid')
    lf = min(len(t1_avg), len(t2_avg), len(gt_avg))
    t1_avg = t1_avg[:lf]
    t2_avg = t2_avg[:lf]
    gt_avg = gt_avg[:lf]

    fig = plt.figure(figsize=(8.75, 3.25)) 
    gs = gridspec.GridSpec(1, 1, height_ratios=[1]) 
    ax0 = plt.subplot(gs[0])
    fig.tight_layout()
    axes = [ax0]#, ax1]
    a = plt.gca()
    axes[0].set_ylim(ymin = 0, ymax = 6)
    axes[0].set_xlim([-0.25,16])

    t1_min = np.divide(range(len(t1_avg)), 60)
    t2_min = np.divide(range(len(t2_avg)), 60)
    gt_min = np.divide(range(len(gt_avg)), 60)

    colors = ['#232D4B', '#64a644', '#E57200']

    axes[0].plot(gt_min, gt_avg, label=labels[2], color=colors[2],alpha=0.75,linewidth=1.82)
    axes[0].plot(t1_min, t1_avg, label=labels[0], color=colors[0],alpha=0.8,linewidth=1.82)
    axes[0].plot(t2_min, t2_avg, label=labels[1], color=colors[1],alpha=0.8,linewidth=1.82)

    # Set labels
    solo_metrics = [MBE(gt_avg,t1_avg), np.sqrt(mean_squared_error(gt_avg,t1_avg))]
    mav_metrics = [MBE(gt_avg,t2_avg), np.sqrt(mean_squared_error(gt_avg,t2_avg))]


    if show_metrics:
        metric_label = ('Solo RMSE:   %.02f m s$^{-1}$  Solo MBE:   %.02f m s$^{-1}$\n'\
        'Mavic RMSE: %.02f m s$^{-1}$ Mavic MBE: %.02f m s$^{-1}$' % (solo_metrics[1], solo_metrics[0], mav_metrics[1], mav_metrics[0]))
        axes[0].annotate(metric_label, 
        (0.01, 0.965),
        xytext=(4, -4),
        xycoords='axes fraction',
        textcoords='offset points',
        fontweight='bold',
        color='k',
        backgroundcolor='white',
        ha='left', va='top')

    #axes[0].set_title("%d training split - %s" % (ti * 100, t))
    axes[0].set_ylabel(y_label, fontsize=18)
    #a.yaxis.set_label_coords(-0.05,-0.1)
    ax0.set_xlabel("Time (minutes)", fontsize=18)
    axes[0].legend(loc="upper right", ncol=1,frameon=True, prop={'size': 13})
    ax0.yaxis.set_ticks(np.arange(0, y_lims[1], 1))
    ax0.tick_params(axis='both', which='major', labelsize=12)
    plt.subplots_adjust(top=0.885,bottom=0.194,left=0.1,right=0.94,hspace=0.2,wspace=0.06)

    if save_figs:
        plt.savefig(base_loc +"\\multi_trial_%s.png" % (title))

    plt.grid(linestyle="--")
    if show_figs:
        plt.show()
    
    plt.close()


def buildLSTM(numFeatures):
    numPoints = None
    multi_step_model = tf.keras.models.Sequential()

    # march model 420-180-90d92-42-lr4e-5
    multi_step_model.add(tf.keras.layers.LSTM(420, return_sequences=True, input_shape=(numPoints, numFeatures)))
    multi_step_model.add(tf.keras.layers.LSTM(180, return_sequences=True))
    multi_step_model.add(tf.keras.layers.LSTM(90, return_sequences=True, dropout=0.96)) # dropout used to combat overfitting in one flight during June
    #multi_step_model.add(tf.keras.layers.LSTM(90, return_sequences=True))
    multi_step_model.add(tf.keras.layers.LSTM(48, activation='relu')) 


    #multi_step_model.add(LeakyReLU(alpha=0.05))activation=tf.nn.tanh)
    multi_step_model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.RMSprop(lr=4e-5)
    
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

    multi_step_model.compile(optimizer=optimizer, loss = 'mse')

    return multi_step_model

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

'''
Trains the model on given data
'''
def train(multi_step_model, trainData, trainLabels, testData, testLabels, Title): 

    t = multi_step_model.fit(trainData, trainLabels, epochs=epochs, batch_size=16, validation_data=(testData, testLabels))
    plt.plot(t.history['loss'])
    plt.plot(t.history['val_loss'])
    plt.title(Title + ' train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right',ncol=2)
    if show_figs:
        plt.show()
    plt.close()

'''
Plots the predictions for the given model on the given data
'''
def plot_predictions(multi_step_model, testData, trainLabels, testLabels, Title, training_increment, save_fig):
    for i in range(1):
        fig = plt.figure(figsize=(12, 3)) 
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        axes = [ax0, ax1]

        x, y = testData, testLabels
        step_size = 1
        predictions = multi_step_model.predict(x, steps=step_size)[:,0]
        avg = np.mean(predictions)
        flipped_predictions = []
        for p in predictions:
            flipped_predictions.append(avg + (avg - p))

        #predictions = flipped_predictions
        plt.figure(i)

        a = plt.gca()
        a.set_ylim([0, 9])
        
        bf = len(trainLabels)
        td = len(testData)

        axes[0].plot(range(bf), trainLabels, label='History', color='Orange')
        axes[0].plot(range(bf, bf+td), y, label='True Wspd', color='#c334e3', alpha=0.8)
        axes[0].plot(range(bf, bf+td), predictions, label='Predicted Wspd', color='Navy')
        
        axes[1].plot(range(bf, bf+td), y, label='True Wspd', color='#c334e3',alpha=0.8,linewidth=1.1)
        axes[1].plot(range(bf, bf+td), predictions, label='Predicted Wspd', color='Navy', linewidth=0.8)

        # TrueWspdVar, PredWspdVar, MBE, MSE, RMSE
        metrics = [np.std(y)**2, np.std(predictions)**2, MBE(y,predictions), mean_squared_error(y,predictions),np.sqrt(mean_squared_error(y,predictions))]
        
        print("-----------LSTM (%1d training) ------------\n%s\n" % (training_increment * 100, Title))
        print('{:34s}: {:4.2f}'.format("true wind speed variance", metrics[0]))
        print('{:34s}: {:4.2f}'.format("predicted wind speed variance", metrics[1]))
        print('{:34s}: {:4.2f}'.format("MBE",metrics[2]))
        print('{:34s}: {:4.2f}'.format("MSE",metrics[3]))
        print('{:34s}: {:4.2f}'.format("RMSE",metrics[4]))

        print('{:34s}: {:4.2f}'.format("true turbulence intensity",findTurbulence(y)))
        print('{:34s}: {:4.2f}'.format("predicted turbulence intensity",findTurbulence(predictions)))
        print('{:34s}: {:4.2f}'.format(" turbulence intensity diff ",findTurbulence(y) - findTurbulence(predictions)))

        fig.tight_layout()
        
        # Set labels
        fig.text(0.5, 0.04, 'Time Step', ha='center', va='center')
        fig.text(0, 0.8, 'Wspd', ha='center', va='center', rotation='vertical')

        axes[0].set_title(Title + " - Training + Testing Data (%1f split)" % (training_increment * 10))
        axes[1].set_title(Title + " - Testing Data Only")

        ax0.legend(loc="best")
        ax1.legend(loc="best")
        plt.legend(ncol=2)
        plt.subplots_adjust(top=0.885,bottom=0.194,left=0.06,right=0.94,hspace=0.2,wspace=0.06)
        
        try:
            moving_metrics = plot_moving_avg(list(itertools.chain.from_iterable(y)), predictions, Title, training_increment)
            #moving_metrics = plot_moving_avg_final(np.array(y).ravel(), predictions, Title, training_increment)

        except Exception as e:
            print(e)
            moving_metrics = plot_moving_avg_final(y, predictions, Title, training_increment)
        
        plt.close()
        if use_moving_avg_metrics:
            return moving_metrics
        return metrics

def findTurbulence(ws):
    return round(np.std(ws) / np.mean(ws), 3)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1)) 

'''
Standard Scaling:
Scales data columns by taking each row, subtracting the column mean from it, 
and dividing it by the column standard deviation
'''
def scale(d):
    return (d-np.mean(d))/np.std(d)

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


'''
1D turbulence utility functions
'''
def find_turb_var(u, v):
    u_var = np.var(u)
    v_var = np.var(v)
    print("u turb var: %.03f\nv turb var: %.03f" % (u_var, v_var))

    return u_var, v_var

# convert wind to streamwise coordinates
# (https://www.eol.ucar.edu/content/wind-direction-quick-reference)
# "As expected, if U=Uav and V=Vav then Ustream = Spd, and Vstream = 0."
# where average wind vectors are Uav, Vav
def to_streamwise_coord(u, v, wdir):
    u_stream = []
    v_stream = []
    mean_wind_dir = np.mean(wdir)
    for i in range(len(u)):
        d = np.degrees(np.arctan2(v[i], u[i]))
        u_stream.append(u[i] * np.cos(np.radians(d)) + v[i] * np.sin(np.radians(d)))
        v_stream.append(-1 * u[i] * np.sin(np.radians(d)) + v[i] * np.cos(np.radians(d)))

    return u_stream, v_stream

# http://tornado.sfsu.edu/geosciences/classes/m430/Wind/WindDirection.html
def split_wspd(wspd, wdir):
    u = []
    v = []
    avg_wind_dir = np.mean(wdir)
    for w in range(len(wspd)):
        met_dir = avg_wind_dir + 180
        u.append(-1 * wspd[w] * np.sin((np.pi / 180) * met_dir))
        v.append(-1 * wspd[w] * np.cos((np.pi / 180) * met_dir))
        #print("%.02f (actual wspd) vs. %.02f (comp wspd)" % (wspd[w], (u[w]**2 + v[w]**2)**0.5))

    return u, v

if __name__ == "__main__":
    main()
