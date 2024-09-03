'''Modified Oxford routines for reading in data and making dataframes'''
import warnings
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, accuracy_score

def load_data(datafile, acc_prefix, check = True, ignore_labels = None):
    """Reads the accelerometer data for a participant from a CSV file and returns in a dataframe, removing consecutive NA entries
       at start and end of dataframe.


    Args:
        datafile (str): filename of CSV data to load for a participant
        acc_prefix (str): prefix of the accelerometer to load data for
        check (bool, optional): Toggle to determine if data checking is on. Defaults to True.
        ignore_labels (list of str, optional): Any data with this label will be ignored. Defaults to None.

    Returns:
        pandas.DataFrame: dataframe with time as index, accelerometer data in columns x, y, and z, and label in column annotation
    """

    acc_x = acc_prefix + '_x'
    acc_y = acc_prefix + '_y'
    acc_z = acc_prefix + '_z'

    data = pd.read_csv(
        datafile,
        index_col='global_time', parse_dates=['global_time'],
        usecols=['global_time', acc_x, acc_y, acc_z, 'posture_movement'],
        dtype={acc_x: 'f4', acc_y: 'f4', acc_z: 'f4', 'posture_movement': 'string'},
        keep_default_na=False,
        na_values={acc_x: ['NA'], acc_y: ['NA'], acc_z: ['NA']},
        encoding='utf-8'
    )

    if check:
        time_diffs = pd.Series(data.index).diff().dt.total_seconds() * 1000 # convert to ms
        freq_counts = time_diffs.value_counts().dropna()
        if pd.Series(data.index).is_monotonic_increasing:
            print('Data is monotonically increasing')
            if len(freq_counts) == 1:
                print('The time difference between each row is constant')
            else:
                print('WARNING: The time difference between each row is not constant')
                print(' '.join([f'(Δ: {diff:.1f} ms, count: {count})' for diff, count in freq_counts.items()]))
                mean_diff = time_diffs.dropna().mean()
                std_diff = time_diffs.dropna().std()
                print(f'Average: {mean_diff:.2f} ms Standard deviation: {std_diff:.2f} ms')
        else:
            print('ERROR: Data is not monotonically increasing')
            non_increasing_times = time_diffs[time_diffs <= 0]
            non_increasing_indices = non_increasing_times.index
            non_increasing_indices = non_increasing_indices.tolist()
            print(f'Non-increasing times found at indices {non_increasing_indices}')
            print(' '.join([f'(Δ: {diff:.1f} ms, count: {count})' for diff, count in freq_counts.items()]))

    data.columns = ['x', 'y', 'z', 'annotation']

    # Remove NA entries at start and end of dataframe
    data.reset_index(inplace=True)

    index_of_start = 0
    index_of_end = len(data)

    # Find index of last NA entry in initial run of NA entries
    if data.iloc[0]['annotation'] == 'NA':
        index_of_start = (data['annotation'] != 'NA').idxmax()

    # Find index of first NA entry in final run of NA entries
    if data.iloc[-1]['annotation'] == 'NA':
        index_of_end  = (data['annotation'] != 'NA')[::-1].idxmax() + 1

    data = data.iloc[index_of_start:index_of_end]
    data.set_index('global_time', inplace=True)
    if check:
        na_count = data['annotation'].str.contains('NA').sum()
        if na_count > 0:
            print(f'WARNING: NA appears {na_count} times in label column')

    if ignore_labels:
        last_len = data.shape[0]
        for label in ignore_labels:
            data = data[data['annotation'] != label]
            if check and data.shape[0] != last_len:
                print(f'{last_len - data.shape[0]} rows removed due to {label} data')
                last_len = data.shape[0]

    return data

def make_windows(data, winsec=30, sample_rate=30, dropna=True, drop_impure=False, verbose=False, frame_info=False):
    """Make non-overlapping windows from a dataframe of accelerometer data.

    Args:
        data (pandas.DataFrame): dataframe for a participant with time as index, accelerometer data in columns x, y, and z, and label in column annotation
        winsec (int, optional): desired size of windows in seconds. Defaults to 30.
        sample_rate (int, optional): sample rate of data in Hz. Defaults to 30.
        dropna (bool, optional): switch to determine if nan should be dropped. Defaults to True.
        drop_impure (bool, optional): switch to determine if windows where all labels aren't identical should be dropped. Defaults to False.
        verbose (bool, optional): switch to determine if extra information is printed during window creation. Defaults to False.
        frame_info (bool, optional): switch to determine if frame by frame details are given. Needs verbose to be True to do anything. Defaults to False.

    Returns:
        tuple: A tuple (X, Y, T) of numpy arrays, each described as follows:
        - X (np.ndarray): 3D array of shape (n_windows, winsec*sample_rate, 3) containing the accelerometer data
        - Y (np.ndarray): 1D array of shape (n_windows,) containing the labels
        - T (np.ndarray): 1D array of shape (n_windows,) containing the timestamps
    """

    X, Y, T = [], [], []
    frame = 0
    empty_windows, bad_windows, impure_windows = [], [], []

    for t, w in tqdm(data.groupby(pd.Grouper(freq=f'{winsec}s', origin='start', closed='left', dropna=False)), disable=verbose):
        frame += 1
        if len(w) < 1:  # skip if empty window
            if verbose:
                if frame_info:
                    print(f'Frame {frame} is empty')
                empty_windows.append(frame)
            continue

        t = t.to_numpy()

        x = w[['x', 'y', 'z']].to_numpy()

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unable to sort modes')
            label_counts = w['annotation'].value_counts(dropna=False)
            dominant_label = label_counts.idxmax()
            dominant_label_percentage = (label_counts.max() / len(w['annotation'])) * 100
            if drop_impure:
                if w['annotation'].nunique(dropna = False) == 1:
                    # pure window
                    y = w['annotation'].iloc[0]
                else:
                    # impure window
                    if verbose:
                        if frame_info:
                            if is_good_window(x, t, frame, sample_rate, winsec):
                                print(f'Frame {frame} has {dominant_label_percentage:.2f}% {dominant_label} labels but impure')
                            else:
                                print(f'Frame {frame} has {dominant_label_percentage:.2f}% {dominant_label} labels but impure and not good')
                        impure_windows.append(frame)
                    continue
            else:
                y = w['annotation'].mode(dropna=False).iloc[0]

        if dropna and pd.isna(y):  # skip if annotation is NA
            if verbose:
                print('|Window with NA at frame {frame} dropped')
            continue

        if not is_good_window(x, t, frame, sample_rate, winsec):  # skip if bad window
            if verbose:
                if frame_info:
                    print(f'Frame {frame} has {dominant_label_percentage:.2f}% {dominant_label} labels but not good')
                bad_windows.append(frame)
            continue

        if frame_info:
            print(f'Frame {frame} has {dominant_label_percentage:.2f}% {dominant_label} labels')
        X.append(x)
        Y.append(y)
        T.append(t)

    # edge case when no windows created should be caught
    if len(X) > 0:
        X = np.stack(X)
        Y = np.stack(Y)
        T = np.stack(T)

    if verbose:
        print(f'{frame} windows examined')
        if empty_windows:
            print(f'{len(empty_windows)} empty windows at frames {empty_windows}')
        if bad_windows:
            print(f'{len(bad_windows)} bad windows dropped at frames {bad_windows}')
        if impure_windows:
            print(f'{len(impure_windows)} impure windows dropped at frames {impure_windows}')
        if len(X) > 0:
            print(f'{X.shape[0]} windows created')

    return X, Y, T

def is_good_window(x, t, frame, sample_rate, winsec):
    ''' Check there are no NaNs and len is good '''

    # Check window length is correct
    window_len = sample_rate * winsec
    if len(x) < window_len:
        return False
    elif len(x) > window_len:
        # temporary fix for special case where frame has 1 extra row - hopefully can remove after latest data release
        # when removed can also remove t and frame as arguments
        # error still present in v1.1 so will keep in
        if len(x) == window_len + 1:
            print(f'WARNING: Window length of {len(x)} is greater than the expected value of {window_len} at frame {frame}')
            print(f'Timestamp of start of this window is {t}')
            return False
        else:
            raise ValueError(f'Window length of {len(x)} is greater than the expected value of {window_len}')

    # Check no nans
    if np.isnan(x).any():
        print('NaNs found')
        return False

    return True

def map_to_new_classes(df, column_name, mapping_file, verbose=True):
    """Maps the classes in a specified column of a dataframe to a different set of classes specified in a JSON file.

    Args:
        df (pandas.DataFrame): dataframe containing the data with labels in a column
        column_name (str): name of the column containing the labels
        mapping_file (str): full path of the mapping JSON file
        verbose (bool, optional): switch to print additional information. Defaults to True.

    Returns:
        pandas.DataFrame: The dataframe with the labels in the first column mapped to the new classes
    """

    # Read the class map from the JSON file
    with open(mapping_file, 'r', encoding='utf-8') as file:
        class_map = json.load(file)

    if verbose:
        unique_values = df[column_name].unique()

        # Warn if the class map does not contain all the unique values in Y
        mismatched_values = [value for value in unique_values if value not in class_map]
        if len(mismatched_values) > 0:
            print(f'The class map does not contain the following values: {mismatched_values}')
        # Warn if the class map has values that are not in Y
        mismatched_values = [value for value in class_map if value not in unique_values]
        if len(mismatched_values) > 0:
            print(f'The class map contains the following values that are not the dataset: {mismatched_values}')

    # Map the values in Y using the class map
    df[column_name] = df[column_name].map(class_map)
    return df


def per_participant_metrics(Y_test, Y_test_pred, pid_test):
    """ Calculate metrics on a per participant basis and provide summary statistics

    Args:
    Y_test (array-like): True labels
    Y_test_pred (array-like): Predicted labels
    pid_test (array-like): Participant IDs corresponding to each sample

    Returns:
    str: Formatted report of per-participant metrics and summary statistics
    """

    unique_pids = np.unique(pid_test)
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'kappa': []
    }

    report = f"{'Participant':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Kappa':<12}\n"
    report += "-" * 72 + "\n"

    for participant in unique_pids:
        participant_mask = (pid_test == participant)
        Y_test_participant = Y_test[participant_mask]
        Y_test_pred_participant = Y_test_pred[participant_mask]

        accuracy = accuracy_score(Y_test_participant, Y_test_pred_participant)
        precision = precision_score(Y_test_participant, Y_test_pred_participant, average='macro', zero_division=np.nan)
        recall = recall_score(Y_test_participant, Y_test_pred_participant, average='macro', zero_division=np.nan)
        f1 = f1_score(Y_test_participant, Y_test_pred_participant, average='macro', zero_division=np.nan)
        cohen_kappa = cohen_kappa_score(Y_test_participant, Y_test_pred_participant)

        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['kappa'].append(cohen_kappa)

        report += f"{participant:<12} {accuracy:.3f}        {precision:.3f}        {recall:.3f}        {f1:.3f}        {cohen_kappa:.3f}\n"

    report += "-" * 72 + "\n"
    report += "Average      "

    for metric, values in metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        report += f"{mean:.3f}±{std:.3f}  "

    return report