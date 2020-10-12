import numpy as np
from argparse import ArgumentParser
from GradientDescent import gd
from StochasticGradientDescent import sgd
from NewtonRaphson import nr
from logits import plotLL, predict
from data import X_digits, X_digits_test, X_news, X_news_test, y_digits, y_digits_test, y_news, y_news_test


def main(args):

    datasets = ["news", "digits"]
    optimizations = {"gd":gd, "sgd":sgd, "nr":nr}

    lr = args.lr
    iterations = args.iterations
    dataset = args.data
    optim = args.optim

    if dataset not in datasets:
        print(f"ERROR: -d must be within {datasets}")
        exit(-1)

    if optim not in optimizations:
        print(f"ERROR: -optim must be within {optimizations.keys()}")
        exit(-1)

    # train on news dataset
    if dataset == datasets[0]:
        n_cls = len(np.unique(y_news))
        n_feats = X_news.shape[1]
        X = X_news
        X_test = X_news_test
        y = y_news
        y_test = y_news_test

    # train on digits dataset
    else:
        n_cls = len(np.unique(y_digits))
        n_feats = X_digits.shape[1]
        X = X_digits
        X_test = X_digits_test
        y = y_digits
        y_test = y_digits_test

    # init weights and biases
    W = np.random.rand(n_cls, n_feats)
    biases = np.random.rand(n_cls, 1)

    optimizer = optimizations[optim]

    W, biases, LL = optimizer(X, y, W, biases, lr, n_cls, iterations)

    # predict test accuracy
    acc = predict(X_test, y_test, W, biases, n_cls)
    print(f"{dataset} accuracy: {acc:.2f}%")

    # plot log likelihood
    plotLL(LL, iterations, dataset, optim, lr, acc)


if  __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-lr", type=float, default=.01, help="Learning rate")
    parser.add_argument("-it", "--iterations", default=100, type=int, help="Number of iterations to train on")
    parser.add_argument("-d", "--data", type=str, required=True, help="Dataset to train on (digits, news)")
    parser.add_argument("-o", "--optim", type=str, required=True, help="Optimization algorithm to utilize")
    args = parser.parse_args()

    main(args)
