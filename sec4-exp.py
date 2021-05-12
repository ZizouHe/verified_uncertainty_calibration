import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from utils import *
from calibrators import *
import pickle
import argparse
from functools import partial

def compare_exp(data_path, n1, times, mse_path,
                ce_path, cifar=True, top=True):
    """
    compare histogram and scaling_binning methods

    Args
    ----
    data_path: data file path
    n1: calibration dataset size
    times: number of trials
    mse_path: mse vs calibration error plot save path
    ce_path: calibration error plot save path
    cifar: cifar/imagenet
    top: top/marginal calibration
    """
    np.random.seed(0)
    prob, label = pickle.load(open(data_path, "rb"))
    ce_list, stddev_list, mse_list = [], [], []
    if top:
        calib_method = (histogram_top, partial(scale_bin_top, scaler=platt_scaler))
        evaluator = ce_top
        eval_mse = eval_mse_top
    else:
        calib_method = (histogram_margin, partial(scale_bin_margin, scaler=platt_scaler))
        evaluator = ce_marginal
        eval_mse = eval_mse_marginal
    if cifar:
        bin_list = list(range(10, 101, 10))
    else:
        bin_list = list(range(20, 201, 20))

    for num_bins in bin_list:
        print("number of bins: {}".format(num_bins))
        l2_ces, mses = [], []
        # number of trials
        for i in range(times):
            # resample
            indices = np.random.choice(
                    list(range(len(prob))),
                    size=n1, replace=True)
            calib_prob = np.array([prob[i] for i in indices])
            calib_label = np.array([label[i] for i in indices])
            rec_ce, rec_mse = [], []
            # platt binning and histogram
            for method in calib_method:
                calibrator = method(calib_prob, calib_label, num_bins)
                calibrated_prob = calibrator(prob)
                # record performance
                rec_ce.append(evaluator(calibrated_prob, prob, label)**2)
                rec_mse.append(eval_mse(calibrated_prob, prob, label))
            l2_ces.append(rec_ce)
            mses.append(rec_mse)

        # calculate performance
        ce_list.append(np.mean(l2_ces, axis=0))
        stddev_list.append(np.std(l2_ces, axis=0) / np.sqrt(times))
        mse_list.append(np.mean(mses, axis=0))

    l2_ces, l2_stddevs, mses = (np.transpose(ce_list),
                                np.transpose(stddev_list),
                                np.transpose(mse_list))

    # plot
    plot_mse_ce_curve(bin_list, l2_ces, mses, save_path=mse_path)
    plot_ces(bin_list, l2_ces, l2_stddevs,
             labels=["histogram","scaling-binning"], save_path=ce_path)


def compare_exp2(data_path, n1, times, ce_path, cifar=True, top=True):
    """
    compare different scalers in scaling_binning method:
        platt scaler, temperature scaler, dirichlet scaler

    Args
    ----
    data_path: data file path
    n1: calibration dataset size
    times: number of trials
    ce_path: calibration error plot save path
    cifar: cifar/imagenet
    top: top/marginal calibration
    """
    np.random.seed(0)
    prob, label = pickle.load(open(data_path, "rb"))
    ce_list, stddev_list = [], []
    if top:
        calib_method = (partial(scale_bin_top, scaler=platt_scaler),
                        partial(scale_bin_top, scaler=temperature_scaler),
                        partial(scale_bin_top, scaler=dirichlet_scaler))
        evaluator = ce_top
    else:
        calib_method = (partial(scale_bin_margin, scaler=platt_scaler),
                        partial(scale_bin_margin, scaler=temperature_scaler),
                        partial(scale_bin_margin, scaler=dirichlet_scaler))
        evaluator = ce_marginal
    if cifar:
        bin_list = list(range(10, 101, 10))
    else:
        bin_list = list(range(20, 201, 20))

    for num_bins in bin_list:
        print("number of bins: {}".format(num_bins))
        l2_ces = []
        # number of trials
        for i in range(times):
            # resample
            indices = np.random.choice(
                    list(range(len(prob))),
                    size=n1, replace=True)
            calib_prob = np.array([prob[i] for i in indices])
            calib_label = np.array([label[i] for i in indices])
            rec_ce = []
            # platt binning and histogram
            for method in calib_method:
                calibrator = method(calib_prob, calib_label, num_bins)
                calibrated_prob = calibrator(prob)
                # record performance
                rec_ce.append(evaluator(calibrated_prob, prob, label)**2)
            l2_ces.append(rec_ce)

        ce_list.append(np.mean(l2_ces, axis=0))
        stddev_list.append(np.std(l2_ces, axis=0) / np.sqrt(times))

    l2_ces, l2_stddevs = (np.transpose(ce_list), np.transpose(stddev_list))
    
    # plot
    plot_ces(bin_list, l2_ces, l2_stddevs,
             labels=["platter","temperature","dirichlet"], save_path=ce_path)


def plot_ces(bin_list, l2_ces, l2_stddevs, labels, save_path):
    """
    calibration error plot

    Args
    ----
    bin_list: list of number of bins
    l2_ces: l2 calibration error list
    l2_stddevs: standard deviation of l2 calibration error
    labels: labels of legend
    save_path: figure save path
    """
    plt.clf()
    font = {'size': 16}
    rc('font', **font)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    error_bars_90 = 1.645 * l2_stddevs
    colors = ["red", "blue", "green", "black"]
    for i, label in enumerate(labels):
        plt.errorbar(
          bin_list, l2_ces[i], yerr=[error_bars_90[i], error_bars_90[i]],
          barsabove=True, color=colors[i], capsize=4, label=label, linestyle='--')
    plt.ylabel("Squared Calibration Error")
    plt.xlabel("Number of Bins")
    plt.ylim(bottom=0.0)
    plt.legend(loc='lower right')
    #plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


def plot_mse_ce_curve(bin_list, l2_ces, mses, save_path, xlim=None, ylim=None):
    """
    mse vs calibration error plot

    Args
    ----
    bin_list: list of number of bins
    l2_ces: l2 calibration error list
    mses: mse list
    save_path: figure save path
    """
    plt.clf()
    font = {'size': 16}
    rc('font', **font)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    def get_pareto_points(data):
        pareto_points = []
        def dominated(p1, p2):
            return p1[0] >= p2[0] and p1[1] >= p2[1]
        for datum in data:
            num_dominated = sum(map(lambda x: dominated(datum, x), data))
            if num_dominated == 1:
                pareto_points.append(datum)
        return pareto_points
    print(get_pareto_points(list(zip(l2_ces[0], mses[0], bin_list))))
    print(get_pareto_points(list(zip(l2_ces[1], mses[1], bin_list))))
    l2ces0, mses0 = zip(*get_pareto_points(list(zip(l2_ces[0], mses[0]))))
    l2ces1, mses1 = zip(*get_pareto_points(list(zip(l2_ces[1], mses[1]))))
    plt.scatter(l2ces0, mses0, c='red', marker='o', label='histogram')
    plt.scatter(l2ces1, mses1, c='blue', marker='x', label='scaling-binning')
    plt.legend(loc='upper right')
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel("Squared Calibration Error")
    plt.ylabel("Mean-Squared Error")
    plt.tight_layout()
    plt.savefig(save_path)

if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        default='../data/cifar_probs_vgg16.dat', type=str)
    parser.add_argument('--n1', default=1000, type=int)
    parser.add_argument('--ce_fig',
                        default='../pic/cifar_vgg16_top_ce.png', type=str)
    parser.add_argument('--mse_fig',
                        default='../pic/cifar_vgg16_top_mse.png', type=str)
    parser.add_argument('--times', default=100, type=int, help='evaluation times')
    parser.add_argument('--cifar', default=False, action='store_true')
    parser.add_argument('--top', default=False, action='store_true')
    parser.add_argument('--exp', default=1, type=int)
    args = parser.parse_args()
    # run exp1 or exp2
    if args.exp == 1:
        compare_exp(args.data_path, args.n1, args.times, args.mse_fig,
                    args.ce_fig, cifar=args.cifar, top=args.top)
    else:
        compare_exp2(args.data_path, args.n1, args.times,
                     args.ce_fig, cifar=args.cifar, top=args.top)

