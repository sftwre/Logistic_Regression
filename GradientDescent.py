import numpy as np
from argparse import ArgumentParser
from logits import linearPredict, softmax, cross_entropy, predict, plotLL
from data import X_digits, X_digits_test, X_news, X_news_test, y_digits, y_digits_test, y_news, y_news_test


def gd(X: np.ndarray, y: np.ndarray, W: np.ndarray,
       biases: np.ndarray, lr: float, n_cls: int,
       iterations: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Gradient descent
    """

    n = len(y)

    L2 = np.sum(1 / n * W)

    axis0 = np.arange(X.shape[0])

    cost_history = np.zeros(iterations)

    for it in range(iterations):

        logitScores = linearPredict(X, W, biases)
        probs = softmax(logitScores, n_cls)

        # error
        cost_history[it] = cross_entropy(probs, y, W)

        # compute error for probability of correct outcome
        probs[axis0, y.squeeze(-1)] -= 1

        # gradient of weights and biases
        gradsW = probs.T.dot(X) + L2
        gradsBiases = np.sum(probs, axis=0).reshape(-1, 1)

        # update weights
        W -= (lr * gradsW)

        # update biases
        biases -= (lr * gradsBiases)

    return W, biases, cost_history


def main(args):

    datasets = ["news", "digits"]

    lr = args.lr
    iterations = args.iterations
    dataset = args.data

    if dataset not in datasets:
        print(f"ERROR: -d must be within {datasets}")
        exit(-1)

    # load news dataset and run gd
    if dataset == datasets[0]:
        n_cls_news = len(np.unique(y_news))
        n_feats_news = X_news.shape[1]

        W_news = np.random.rand(n_cls_news, n_feats_news)
        biases_news = np.random.rand(n_cls_news, 1)

        W_news, biases_news, LL_news = gd(X_news, y_news, W_news, biases_news, lr, n_cls_news, iterations)

        # predict test accuracy
        acc = predict(X_news_test, y_news_test, W_news, biases_news, n_cls_news)
        print(f"News accuracy: {acc:.2f}%")

        # plot log likelihood
        plotLL(LL_news, iterations, "news", "GD", lr, acc)


    else:
        n_cls_digits = len(np.unique(y_digits))
        n_feats_digits = X_digits.shape[1]

        W_digits = np.random.rand(n_cls_digits, n_feats_digits)
        biases_digits = np.random.rand(n_cls_digits, 1)

        # train
        W_digits, biases_digits, LL_digits = gd(X_digits, y_digits, W_digits, biases_digits, lr, n_cls_digits, iterations)

        # predict test accuracy
        acc = predict(X_digits_test, y_digits_test, W_digits, biases_digits, n_cls_digits)
        print(f"Digits accuracy: {acc:.2f}%")

        # plot log likelihood
        plotLL(LL_digits, iterations, "digits", "GD", lr, acc)


if  __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-lr", type=float, default=.01, help="Learning rate")
    parser.add_argument("-it", "--iterations", default=100, type=int, help="Number of iterations to train on")
    parser.add_argument("-d", "--data", type=str, required=True, help="Dataset to train on (digits, news)")
    parser.add_argument("-o", "--optim", type=str, help="Optimization algorithm to utilize")
    args = parser.parse_args()

    main(args)
