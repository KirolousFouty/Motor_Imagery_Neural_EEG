import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn import datasets, linear_model

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def relative_change(data, pre_onset_data):
    return (data - pre_onset_data) / pre_onset_data

def CAR_filter(data):
    rowAverages = np.mean(data, axis=1)
    rowAverages = np.expand_dims(rowAverages, axis=1)
    rowAverages = np.tile(rowAverages, (1, 15))
    CAR_filtered = data - rowAverages
    return CAR_filtered



def process_subject(signals_file, filtered_file, labels_file, trial_file, fs):
    # read signals, labels, and trial data from files
    signals = pd.read_csv(signals_file, header=None).values
    labels = pd.read_csv(labels_file, header=None).values.squeeze()
    trials = pd.read_csv(trial_file, header=None).values.squeeze()

    # apply CAR filter
    car_filtered = CAR_filter(signals)
    # export csv files to be used for plotting
    np.savetxt(filtered_file, car_filtered, delimiter=',')



    # initializing best combination
    min_error = float('inf')
    best_electrode = None
    best_band = None
    best_K = None

    # Iterate over electrodes, frequency bands, and values of K
    for electrode in range(signals.shape[1]):
        for band in ['Mu', 'Beta']:
            for K in range(1, 11):
                # bandpass filter
                if band == 'Mu':
                    filtered = bandpass_filter(signals[:, electrode], 8, 12, fs)
                else:  # band == 'Beta'
                    filtered = bandpass_filter(signals[:, electrode], 16, 24, fs)

                # apply KNN
                knn = KNeighborsClassifier(n_neighbors=K, metric='euclidean')
                
                # splitting the data for later evaluation
                split_index = int(len(filtered) * 0.8)
                filtered_train = np.resize(filtered[:split_index], (split_index, 15))
                filtered_test = np.resize(filtered[split_index:], (len(filtered) - split_index, 15))
                


                labels_train = np.zeros_like(filtered_train)
                i = 0
                for trial in trials:
                    if trial >= split_index:
                        continue
                    labels_train[int(trial)] = labels[i]
                    i += 1

                labels_test = labels[split_index:]
                labels_test = np.zeros_like(filtered_test)
                for trial in trials:
                    if trial < split_index:
                        continue
                    labels_test[int(trial)-split_index] = labels[i]
                    i += 1


                # training
                # print(f"filtered_train size: {filtered_train.shape}")
                # print(f"labels_train size: {labels_train.shape}")
                knn.fit(filtered_train, labels_train)

                # evaluation
                # print(f"filtered_test size: {filtered_test.shape}")
                # print(f"labels_test size: {labels_test.shape}")                
                labels_pred = knn.predict(filtered_test)
                error = 0
                error += np.mean(labels_pred != labels_test)
                print (error / 10)

                # updating the best combination
                if error < min_error:
                    min_error = error
                    best_electrode = electrode
                    best_band = band
                    best_K = K

    # Output
    print(f"Best Electrode: {best_electrode + 1}, Best Band: {best_band}, Best K: {best_K}, Min Error: {min_error}")


# main
for i in range(1, 4):
    signals_file = f"Subject{i}_Signals.csv"
    filtered_file = f"Subject{i}_Filtered.csv"
    labels_file = f"Subject{i}_Labels.csv"
    trial_file = f"Subject{i}_Trial.csv"
    fs = 512
    print()
    print(f"Subject {i}")
    process_subject(signals_file, filtered_file, labels_file, trial_file, fs)
