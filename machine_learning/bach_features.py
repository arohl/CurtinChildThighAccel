"""Features for the accelerometer data taken from Bach et al, JMPB"""
from scipy import stats
from scipy import fftpack
import numpy as np

def zero_crossing_rate(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum()

def total_signal_energy(signal):
    return np.sum(signal**2)

def frequency_domain_features(signal, sample_rate):
    # Perform the Fast Fourier Transform (FFT)
    spectrum = fftpack.fft(signal)
    # Calculate the absolute values of the FFT (to get the magnitude)
    magnitudes = np.abs(spectrum) / len(signal)
    # Calculate the frequencies for the spectrum
    frequencies = fftpack.fftfreq(len(signal), 1 / sample_rate)
    # Only consider the positive frequencies (the spectrum is symmetric)
    magnitudes = magnitudes[frequencies >= 0]
    frequencies = frequencies[frequencies >= 0]
    # Calculate the mean amplitude and standard deviation
    mean_amp = np.mean(magnitudes)
    sd = np.std(magnitudes)
    # Calculate the cumulative distribution of the power spectrum
    cumulative_magnitudes = np.cumsum(magnitudes)
    # Find the frequency at which half the total power is reached
    median_freq = frequencies[np.searchsorted(cumulative_magnitudes, cumulative_magnitudes[-1] / 2)]
    return mean_amp, sd, median_freq

def bach_features(xyz, sample_rate, features_to_keep=None):

    feats = {}
    quartiles = [0, 25, 50, 75, 100]  # Percentiles for quartiles

    # time domain features
    for i, axis in enumerate('xyz'):
        signal = xyz[:, i]
        feats[f'{axis}Mean'] = np.mean(signal)
        feats[f'{axis}Std'] = np.std(signal)
        feats[f'{axis}Min'], feats[f'{axis}Q25'], feats[f'{axis}Med'], feats[f'{axis}Q75'], feats[f'{axis}Max'] = np.percentile(signal, quartiles)
        if feats[f'{axis}Std'] > 0.01:
            feats[f'{axis}Skew'] = np.nan_to_num(stats.skew(signal))
            feats[f'{axis}Kurt'] = np.nan_to_num(stats.kurtosis(signal))
        else:
            feats[f'{axis}Skew'] = feats[f'{axis}Kurt'] = 0
        feats[f'{axis}ZCR'] = zero_crossing_rate(signal)
        feats[f'{axis}TotalE'] = total_signal_energy(signal)

    # correlation features
    x, y, z = xyz.T
    with np.errstate(
        divide='ignore', invalid='ignore'
    ):  # ignore div by 0 warnings
        feats['xyCorr'] = np.nan_to_num(np.corrcoef(x, y)[0, 1])
        feats['yzCorr'] = np.nan_to_num(np.corrcoef(y, z)[0, 1])
        feats['zxCorr'] = np.nan_to_num(np.corrcoef(z, x)[0, 1])


    for i, axis in enumerate('xyz'):
        signal = xyz[:, i]
        feats[f'{axis}MeanFreqAmp'], feats[f'{axis}SDFreq'], feats[f'{axis}MedianFreq'] = frequency_domain_features(signal, sample_rate)
    
    if features_to_keep:
        feats = {k: feats[k] for k in feats.keys() if k in features_to_keep} # keep kept features in same order

    return feats
