import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from utils import *
from calibrators import *
import pickle

def bootstrap_uncertainty(data, func, alpha=10.0, times=1000):
    """
    boostrap uncertainty estiamtion

    Args
    ----
    data: numpy array
    func: evaluation method
    alpha: 1-alpha confidence interval
    times: number of boostrap times
    """
    est = func(data)
    result = []
    for _ in range(times):
        indices = np.random.choice(list(range(len(data))), size=len(data), replace=True)
        resample_data = [data[i] for i in indices]
        result.append(func(resample_data))
    return (est*2-np.percentile(result, 100 - alpha / 2.0),
            est*2-np.percentile(result, 50),
            est*2-np.percentile(result, alpha / 2.0))


def sec3_experiment(prob, label, n1, n2, bin_func, bins_list, fig_name,
                    times=1000, power=2):
    """
    experiment: binning underestimates calibration error

    Args
    ----
    prob: uncalibrated probability vector, nxk numpy array
    label: classification label, nx1 numpy array
    n1: calibration dataset size
    n2: binning dataset size
    bin_func: binning scheme
    bins_list: list of number of bins
    fig_name: figure save path
    times: bootstrap simulation times
    power: l1/l2 calibration error
    """
    np.random.seed(0)
    indices = np.random.choice(np.arange(prob.shape[0]), size=prob.shape[0], replace=False)
    prob = prob[indices]
    label = label[indices]
    pred = np.argmax(prob, 1)
    prob = np.max(prob, 1)
    correct = (pred == label).astype(dtype=np.int64)
    # train platt scaler on n1
    platt = platt_scaler(prob[:n1], correct[:n1])
    platt_prob = platt(prob)

    lower, middle, upper = [], [], []
    for num_bins in bins_list:
        # binning on n2
        bins = bin_func(platt_prob[:n1+n2], num_bins=num_bins)
        # test on n3
        test_prob = platt_prob[n1+n2:]
        test_correct = correct[n1+n2:]
        test_data = list(zip(test_prob, test_correct))
        def estimator(data):
            binned_data = get_binned_data(data, bins)
            return plugin_ce(binned_data, power=power)
        print('estimate: ', estimator(test_data))
        # bootstrap
        bootstrap_interval = bootstrap_uncertainty(test_data, estimator, times=times)
        if True:
            bootstrap_interval = np.sort(np.abs(np.array(bootstrap_interval)))
        lower.append(bootstrap_interval[0])
        middle.append(bootstrap_interval[1])
        upper.append(bootstrap_interval[2])
        print('interval: ', bootstrap_interval)

    # plot bootstrap results
    lower_errors = np.array(middle) - np.array(lower)
    upper_errors = np.array(upper) - np.array(middle)
    plt.clf()
    font = {'size': 18}
    rc('font', **font)
    plt.errorbar(
        bins_list, middle, yerr=[lower_errors, upper_errors],
        barsabove=True, fmt = 'none', color='black', capsize=4)
    plt.scatter(bins_list, middle, color='black')
    plt.xlabel(r"No. of bins")
    plt.ylabel("l%d Calibration error" % power)
    plt.xscale('log', basex=2)
    #plt.yscale('log', basey=2)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.tight_layout()
    plt.savefig(fig_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/cifar_probs_vgg16.dat', type=str)
    parser.add_argument('--n1', default=1000, type=int)
    parser.add_argument('--n2', default=1000, type=int)
    parser.add_argument('--fig_name', default='../pic/cifar_vgg16_l2.png', type=str)
    parser.add_argument('--bin_func', default='equal_bins', type=str)
    parser.add_argument('--pow', default=2, type=int)
    parser.add_argument('--bins_list', default="2, 4, 8, 16, 32, 64",
                        type=lambda s: [int(t) for t in s.split(',')])
    parser.add_argument('--times', default=1000, type=int,
                        help='bootstrap times')
    args = parser.parse_args()
    # read data
    # pickle.dump((prob, label), open("cifar_probs.dat", "wb"))
    prob, label = pickle.load(open(args.data_path, "rb"))
    if args.bin_func == 'equal_bins':
        bin_func = equal_num_bins
    else:
        bin_func =equal_prob_bins
    # experiment
    sec3_experiment(prob, label, args.n1, args.n2, bin_func,
        args.bins_list, args.fig_name, args.times, args.pow)
