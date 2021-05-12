import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle
import argparse
from functools import partial
from matplotlib.ticker import PercentFormatter
from utils import *
from calibrators import *

def debiased_vs_plugin(data_path, n1, num_bins, times, save_prefix, mode="top", n_initial=2000,
                       n_max=5000, n_incre=500, calibrator=scale_bin_top, power=2):
    """
    experiment that compare debiased estimator versus plugin estimator

    Args
    ----
    data_path: data file path
    n1: calibration dataset size for calibrator
    num_bins: number of bins
    times: number of trials
    save_prefix: plot save path prefix
    mode: "top"/"marginal"
    n_initial: resample data size initial
    n_max: resample data size maximum
    n_incre: resample data size increment
    calibrator: calibration method we use
    power: lp
    """
    np.random.seed(0)
    prob, label = pickle.load(open(data_path, "rb"))
    n, k = prob.shape
    # shuffle data
    indices = np.random.choice(np.arange(n), size=n, replace=False)
    prob = prob[indices]
    label = label[indices]
    pred = np.argmax(prob, 1)
    # calibrator
    calib = calibrator(prob[:n1], label[:n1], num_bins)
    ver_prob = prob[n1:]
    ver_calib_prob = calib(ver_prob)
    ver_label = label[n1:]
    ver_size = list(range(n_initial, 1+np.min([n_max, len(ver_prob)]), n_incre))
    # ce estimator
    ce_estimators = [partial(calibration_error, power=power, mode=mode, debiased=False),
                     partial(calibration_error, power=power, mode=mode, debiased=True)]
    record = np.zeros((len(ce_estimators), len(ver_size), times))
    for idx, size in enumerate(ver_size):
        print("verification dataset size: {}".format(size))
        for i in range(times):
            # resample
            indices = np.random.choice(np.arange(len(ver_prob)), size=size, replace=True)
            sim_ver_prob = ver_prob[indices]
            sim_ver_calib_prob = ver_calib_prob[indices]
            sim_ver_label = ver_label[indices]
            for j in range(len(ce_estimators)):
                record[j][idx][i] = ce_estimators[j](sim_ver_calib_prob, sim_ver_prob, sim_ver_label)**power
    # sort
    record = np.sort(record, axis=-1)
    # ground truth
    truth = np.min([ce_estimators[0](ver_calib_prob, ver_prob, ver_label)**power,
                    ce_estimators[1](ver_calib_prob, ver_prob, ver_label)**power])
    errors = np.abs(record - truth)
    # plot curves and histograms
    plot_mse_curve(errors, ver_size, n1, save_prefix, num_bins, power, mode)
    plot_histograms(errors, n1, save_prefix, num_bins, power, mode)


def plot_mse_curve(errors, ver_size, n1, save_prefix, num_bins, power, mode):
    plt.clf()
    errors = np.square(errors)
    accumulated_errors = np.mean(errors, axis=-1)
    error_bars_90 = 1.645 * np.std(errors, axis=-1) / np.sqrt(n1)
    #print(accumulated_errors)
    plt.errorbar(
        ver_size, accumulated_errors[0], yerr=[error_bars_90[0], error_bars_90[0]],
        barsabove=True, color='red', capsize=4, label='plugin')
    plt.errorbar(
        ver_size, accumulated_errors[1], yerr=[error_bars_90[1], error_bars_90[1]],
        barsabove=True, color='blue', capsize=4, label='debiased')
    plt.ylabel("MSE of Calibration Error")
    plt.xlabel("Number of Samples")
    plt.legend(loc='upper right')
    plt.tight_layout()
    save_name = save_prefix + "plot_" + mode + "_" + str(num_bins) + "_l" + str(power)
    plt.ylim(bottom=0.0)
    plt.savefig(save_name)


def plot_histograms(errors, n1, save_prefix, num_bins, power, mode):
    plt.clf()
    plt.ylabel("Number of estimates")
    plt.xlabel("Absolute deviation from ground truth")
    bins = np.linspace(np.min(errors[:, 0, :]), np.max(errors[:, 0, :]), 40)
    plt.hist(errors[0][0], bins, alpha=0.5, label='plugin')
    plt.hist(errors[1][0], bins, alpha=0.5, label='debiased')
    plt.legend(loc='upper right')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=n1))
    plt.tight_layout()
    save_name = save_prefix + "hist_" + mode + "_" + str(num_bins) + "_l" + str(power)
    plt.savefig(save_name)


if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        default='../data/cifar_probs_vgg16.dat', type=str)
    parser.add_argument('--prefix', default='../pic/cifar_vgg16_debiased_', type=str)
    parser.add_argument('--n1', default=3000, type=int)
    parser.add_argument('--n_initial', default=2000, type=int)
    parser.add_argument('--n_max', default=5000, type=int)
    parser.add_argument('--n_incre', default=500, type=int)
    parser.add_argument('--num_bins', default=15, type=int)
    parser.add_argument('--pow', default=2, type=int)
    parser.add_argument('--times', default=1000, type=int, help='evaluation times')
    parser.add_argument('--top', default=False, action='store_true')
    args = parser.parse_args()
    if args.top:
        mode = "top"
        calibrator = scale_bin_top
    else:
        mode = "marginal"
        calibrator = scale_bin_margin

    debiased_vs_plugin(args.data_path, args.n1, args.num_bins, args.times, args.prefix,
                       mode=mode, n_initial=args.n_initial,  n_max=args.n_max,
                       n_incre=args.n_incre, calibrator=calibrator, power=args.pow)
