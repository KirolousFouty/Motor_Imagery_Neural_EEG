import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

#  apply bandpass filter to data
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

#  calculate relative change between data and pre-onset data
def relative_change(data, pre_onset_data):
    return (data - pre_onset_data) / pre_onset_data

#  apply common average reference (CAR) filter to data
def CAR_filter(data):
    avg = np.mean(data, axis=0)
    CAR_filtered = data - avg
    return CAR_filtered

#  train k-nearest neighbors classifier
def knn_classifier(X, y, K):
    # def
    print()


#  calculate 10-fold classification error
def calc_10_fold_classification_error(X, y, K):
    # def
    print()

#  process subject's data
def process_subject(signals_file, labels_file, trial_file, fs):
    # read signals, labels, and trial data from files
    signals = pd.read_csv(signals_file, header=None).values
    labels = pd.read_csv(labels_file, header=None).values.squeeze()
    trials = pd.read_csv(trial_file, header=None).values.squeeze()

    # apply CAR filter to signals
    car_filtered = CAR_filter(signals)

    # Initialize variables for best combination
    min_error = float('inf')
    best_electrode = None
    best_band = None
    best_K = None

    # Iterate over electrodes, frequency bands, and values of K
    #def

    # Output
    #print(f"Best Electrode: {best_electrode + 1}, Best Band: {best_band}, Best K: {best_K}, Min Error: {min_error}")

# Process each subject's data
for i in range(1, 4):
    signals_file = f"Subject{i}_Signals.csv"
    labels_file = f"Subject{i}_Labels.csv"
    trial_file = f"Subject{i}_Trial.csv"
    fs = 512
    process_subject(signals_file, labels_file, trial_file, fs)
