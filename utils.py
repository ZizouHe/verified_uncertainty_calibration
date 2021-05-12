import numpy as np
from functools import partial

# bin methods

def equal_num_bins(prob, num_bins):
    """
    Get bins that contain approximately an equal number of data points.

    Args
    ----
    prob: probability vector, nx1 numpy array
    num_bins: number of bins
    """
    sorted_prob = sorted(prob)
    k = int(np.ceil(len(sorted_prob) * 1.0 / num_bins))
    binned_prob = [sorted_prob[i:i + k] for i in range(0, len(sorted_prob), k)]
    bins = []
    for i in range(len(binned_prob) - 1):
        last_prob = binned_prob[i][-1]
        next_first_prob = binned_prob[i + 1][0]
        bins.append((last_prob + next_first_prob) / 2.0)
    bins.append(1.0)
    return bins


def equal_prob_bins(probs, num_bins):
    """equal width binning"""
    return [i * 1.0 / num_bins for i in range(1, num_bins + 1)]


def discrete_bins(data):
    """discrete binning, 1 in 1 bin"""
    sorted_values = sorted(np.unique(data))
    bins = []
    for i in range(len(sorted_values) - 1):
        mid = (sorted_values[i] + sorted_values[i+1]) / 2.0
        bins.append(mid)
    bins.append(1.0)
    return bins


def get_bin_probs(binned_data):
    """get binned probability estimates"""
    bin_sizes = list(map(len, binned_data))
    num_data = sum(bin_sizes)
    bin_probs = list(map(lambda b: b * 1.0 / num_data, bin_sizes))
    return list(bin_probs)


def get_binned_data(data, bins):
    """convert data to binned data"""
    data = np.array(data)
    bin_indices = np.searchsorted(bins, data[:, 0])
    bin_sort_indices = np.argsort(bin_indices)
    sorted_bins = bin_indices[bin_sort_indices]
    splits = np.searchsorted(sorted_bins, list(range(1, len(bins))))
    binned_data = np.split(data[bin_sort_indices], splits)
    return binned_data


# calibration error measurement

def plugin_ce(binned_data, power=2):
    """
    calibration error, Kumar et al. (2019) equation 1
    plugin method, Kumar et al. (2019) Def. 5.1

    Args
    ----
    binned_data: data fall into each bin, list of numpy nd array
    power: l1/l2 calibration error
    """
    def bin_error(data):
        if len(data) == 0:
            return 0.0
        return abs(np.mean(data[:, 0])-np.mean(data[:, 1])) ** power
    bin_probs = get_bin_probs(binned_data)
    bin_errors = list(map(bin_error, binned_data))
    return np.dot(bin_probs, bin_errors) ** (1.0 / power)


def debiased_l2_ce(binned_data, power=2):
    """
    calibration error, Kumar et al. (2019) equation 1
    debiased method, Kumar et al. (2019) Def. 5.2

    Args
    ----
    binned_data: data fall into each bin, list of numpy nd array
    """
    def bin_error(data):
        if len(data) < 2:
            return 0.0
        plugin_est = abs(np.mean(data[:, 0])-np.mean(data[:, 1])) ** 2
        label = list(map(lambda x: x[1], data))
        mean_label = np.mean(label)
        debiased_term = mean_label * (1.0 - mean_label) / (len(data) - 1.0)
        return plugin_est - debiased_term
    bin_probs = get_bin_probs(binned_data)
    bin_errors = list(map(bin_error, binned_data))
    return np.max([np.dot(bin_probs, bin_errors), 0.0]) ** 0.5


def normal_debiased_ce(binned_data, power=2):
    """
    calibration error, Kumar et al. (2019) equation 1
    debiased method, Kumar et al. (2019) Def. 5.2

    Args
    ----
    binned_data: data fall into each bin, list of numpy nd array
    """
    bin_sizes = np.array(list(map(len, binned_data)))
    label_means = np.array(list(map(lambda l: np.mean([b for a, b in l]), binned_data)))
    label_stddev = np.sqrt(label_means * (1 - label_means) / bin_sizes)
    model_vals = np.array(list(map(lambda l: np.mean([a for a, b in l]), binned_data)))
    ce = plugin_ce(binned_data, power=power)
    bin_prob = get_bin_probs(binned_data)
    ces = []
    for i in range(1000):
        label_samples = np.random.normal(loc=label_means, scale=label_stddev)
        diff = np.power(np.abs(label_samples - model_vals), power)
        ce = np.power(np.dot(bin_prob, diff), 1.0 / power)
        ces.append(ce)
    return 2*ce - np.mean(ces)


def calibration_error(calib_prob, prob, label, debiased=False,
                      binning_scheme=discrete_bins, mode="top", power=2):
    """
    calculate calibration error

    Args
    ----
    calib_prob: calibrated probability vector, nx1 numpy array
    prob: evaluation probability vector, nx1 numpy array
    label: classification label vector, nx1 numpy array
    debiased: debiased/plugin
    binning_scheme: binning methods
    mode: "top"/"marginal"
    power: lp calibration error
    """
    if debiased and power == 2:
        ce_estimator = debiased_l2_ce
    elif debiased:
        ce_estimator = normal_debiased_ce
    else:
        ce_estimator = plugin_ce
    if mode == "top":
        return ce_top(calib_prob, prob, label, binning_scheme, ce_estimator, power)
    else:
        return ce_marginal(calib_prob, prob, label, binning_scheme, ce_estimator, power)


def ce_top(calib_prob, prob, label, binning_scheme=discrete_bins,
               ce_estimator=plugin_ce, power=2):
    """
    top calibration error, Kumar et al. (2019) equation 2
    plugin method

    Args
    ----
    calib_prob: calibrated probability vector, nx1 numpy array
    prob: evaluation probability vector, nx1 numpy array
    label: classification label vector, nx1 numpy array
    binning_scheme: binning methods
    ce_estimator: which calibration error estimator to use
    power: l1/l2 calibration error
    """
    if len(calib_prob.shape) == 2:
        calib_prob = np.max(calib_prob, 1)
    correct = (np.argmax(prob, 1) == label).astype(dtype=np.int64)
    data = list(zip(calib_prob, correct))
    bins = binning_scheme(calib_prob)
    binned_data = get_binned_data(data, bins)
    return ce_estimator(binned_data, power)


def ce_marginal(calib_prob, prob, label, binning_scheme=discrete_bins,
                    ce_estimator=plugin_ce, power=2):
    """
    marginal calibration error, Kumar et al. (2019) equation 3
    plugin method

    Args
    ----
    calib_prob: calibrated probability vector, nxk numpy array
    prob: evaluation probability vector, nxk numpy array
    label: classification label vector, nx1 numpy array
    binning_scheme: binning methods
    ce_estimator: which calibration error estimator to use
    power: l1/l2 calibration error
    """
    ces = []
    n, k = prob.shape
    label_onehot = np.zeros(prob.shape)
    label_onehot[np.arange(n), label.astype(dtype=np.int64)] = 1
    for i in range(k):
        prob_i = calib_prob[:, i]
        label_i = label_onehot[:, i]
        data_i = list(zip(prob_i, label_i))
        bin_i = binning_scheme(prob_i)
        binned_data_i = get_binned_data(data_i, bin_i)
        ce = ce_estimator(binned_data_i, power) ** power
        ces.append(ce)
    return np.mean(ces)**(1.0/power)


def eval_mse_top(calib_prob, prob, label):
    """top mse error"""
    correct = (np.argmax(prob, 1) == label)
    return np.mean(np.square(calib_prob - correct))


def eval_mse_marginal(calib_prob, prob, label):
    """marginal mse error"""
    n, k = prob.shape
    label_onehot = np.zeros(prob.shape)
    label_onehot[np.arange(n), label.astype(dtype=np.int64)] = 1
    return np.mean(np.square(calib_prob - label_onehot)) * calib_prob.shape[1] / 2.0
