import numpy as np
from sklearn.linear_model import LogisticRegression
import bisect
from utils import *

def platt_scaler(prob, label):
    """
    base method: platt scaler

    Args
    ----
    prob: uncalibrated probability vector, nx1 numpy array
    label: classification label, nx1 numpy array
    """
    model = LogisticRegression(C=1e10, solver='lbfgs')
    eps = 1e-12
    prob = np.expand_dims(prob, axis=-1)
    # clip and transform to logits
    prob = np.clip(prob, eps, 1 - eps)
    prob = np.log(prob / (1 - prob))
    # train model
    model.fit(prob, label)
    # calibration
    def calibrator(x):
        x = np.clip(x, eps, 1 - eps)
        x = np.log(x / (1 - x))
        x = x * model.coef_[0] + model.intercept_
        out = 1 / (1 + np.exp(-x))
        return out
    return calibrator


def temperature_scaler(prob, label):
    """
    base method: temperature scaler

    Args
    ----
    prob: uncalibrated probability vector, nx1 numpy array
    label: classification label, nx1 numpy array
    """
    model = LogisticRegression(fit_intercept=False, C=1e10, solver='lbfgs')
    eps = 1e-12
    prob = np.expand_dims(prob, axis=-1)
    # clip and transform to logits
    prob = np.clip(prob, eps, 1 - eps)
    prob = np.log(prob / (1 - prob))
    # train model
    model.fit(prob, label)
    # calibration
    def calibrator(x):
        x = np.clip(x, eps, 1 - eps)
        x = np.log(x / (1 - x))
        x = x * model.coef_[0]
        out = 1 / (1 + np.exp(-x))
        return out
    return calibrator


def dirichlet_scaler(prob, label):
    """
    base method: dirichlet scaler

    Args
    ----
    prob: uncalibrated probability vector, nx1 numpy array
    label: classification label, nx1 numpy array
    """
    model = LogisticRegression(C=1e10, solver='lbfgs')
    eps = 1e-12
    prob = np.expand_dims(prob, axis=-1)
    # clip and transform to log p
    prob = np.clip(prob, eps, 1 - eps)
    prob = np.log(prob)
    # train model
    model.fit(prob, label)
    # calibration
    def calibrator(x):
        x = np.clip(x, eps, 1 - eps)
        x = np.log(x / (1 - x))
        x = x * model.coef_[0] + model.intercept_
        out = 1 / (1 + np.exp(-x))
        return out
    return calibrator


def histogram_calibrator(prob, values, bins):
    """
    base method: histogram calibrator

    Args
    ----
    prob: probability vector, nx1 numpy array
    values: function value vector, nx1 numpy array
    bins: list of bins
    """
    # group by bins
    binned_values = [[] for _ in range(len(bins))]
    for p, value in zip(prob, values):
        bin_idx = bisect.bisect_left(bins, p)
        binned_values[bin_idx].append(float(value))

    def safe_mean(values, bin_idx):
        if len(values) == 0:
            if bin_idx == 0:
                return float(bins[0]) / 2.0
            return float(bins[bin_idx] + bins[bin_idx - 1]) / 2.0
        return np.mean(values)
    # calculate mean value of each bin
    bin_means = [safe_mean(values, bidx) for values, bidx in zip(binned_values, range(len(bins)))]
    bin_means = np.array(bin_means)
    def calibrator(prob):
        indices = np.searchsorted(bins, prob)
        return bin_means[indices]
    return calibrator


def scale_bin_top(prob, label, num_bins, scaler=platt_scaler):
    """
    scaling binning with top calibraion

    Args
    ----
    prob: uncalibrated probability vector, nxk numpy array
    label: classification label, nx1 numpy array
    num_bins: number of bins
    scaler: base scaler to use
    """
    pred = np.argmax(prob, 1)
    prob = np.max(prob, 1)
    correct = (pred == label).astype(dtype=np.int64)
    # step 1: train on scaler
    func = scaler(prob, correct)
    prob = func(prob)
    # step 2: binning
    bins = equal_num_bins(prob, num_bins)
    bin_calibrator = histogram_calibrator(prob, prob, bins)
    # calibration
    def calibrator(prob):
        return bin_calibrator(func(np.max(prob, 1)))
    return calibrator


def scale_bin_margin(prob, label, num_bins, scaler=platt_scaler):
    """
    scaling binning with marginal calibraion

    Args
    ----
    prob: uncalibrated probability vector, nxk numpy array
    label: classification label, nx1 numpy array
    num_bins: number of bins
    scaler: base scaler to use
    """
    n, k = prob.shape
    # one-hot labeling
    label_onehot = np.zeros(prob.shape)
    label_onehot[np.arange(n), label.astype(dtype=np.int64)] = 1
    funcs = []
    bin_calibrators = []
    # fit 1 vs other
    for i in range(k):
        prob_i = prob[:, i]
        label_i = label_onehot[:, i]
        # step 1: train on scaler
        func = scaler(prob_i, label_i)
        funcs.append(func)
        prob_i = func(prob_i)
        # step 2: binning
        bins = equal_num_bins(prob_i, num_bins)
        bin_calibrator = histogram_calibrator(prob_i, prob_i, bins)
        bin_calibrators.append(bin_calibrator)
    # calibration
    def calibrator(prob):
        calib_prob = np.zeros(prob.shape)
        for i in range(k):
            prob_i = funcs[i](prob[:, i])
            calib_prob[:, i] = bin_calibrators[i](prob_i)
        return calib_prob
    return calibrator


def histogram_top(prob, label, num_bins):
    """
    histogram binning with top calibraion

    Args
    ----
    prob: uncalibrated probability vector, nxk numpy array
    label: classification label, nx1 numpy array
    num_bins: number of bins
    """
    pred = np.argmax(prob, 1)
    prob = np.max(prob, 1)
    correct = (pred == label).astype(dtype=np.int64)
    # binning
    bins = equal_num_bins(prob, num_bins)
    bin_calibrator = histogram_calibrator(prob, correct, bins)

    # calibration
    def calibrator(prob):
        return bin_calibrator(np.max(prob, 1))
    return calibrator


def histogram_margin(prob, label, num_bins):
    """
    histogram binning with marginal calibraion

    Args
    ----
    prob: uncalibrated probability vector, nxk numpy array
    label: classification label, nx1 numpy array
    num_bins: number of bins
    """
    n, k = prob.shape
    # one-hot labeling
    label_onehot = np.zeros(prob.shape)
    label_onehot[np.arange(n), label.astype(dtype=np.int64)] = 1
    bin_calibrators = []
    # fit 1 vs other
    for i in range(k):
        prob_i = prob[:, i]
        label_i = label_onehot[:, i]
        # binning
        bins = equal_num_bins(prob_i, num_bins)
        bin_calibrator = histogram_calibrator(prob_i, label_i, bins)
        bin_calibrators.append(bin_calibrator)

    # calibration
    def calibrator(prob):
        calib_prob = np.zeros(prob.shape)
        for i in range(k):
            calib_prob[:, i] = bin_calibrators[i](prob[:, i])
        return calib_prob
    return calibrator
