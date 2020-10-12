import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from data import X_digits, X_digits_test, X_news, X_news_test, y_digits, y_digits_test, y_news, y_news_test


sns.set_theme(style="darkgrid")

def softmax(logits: np.ndarray, n_cls: int):
    """
    Converts logit scores for output classes to probability distribution.

    logits: Matrix of logit scores for each row of features in the training data.

    returns: Probability distribution over output classes for each feature set.
    """

    probs = np.array([np.empty([n_cls]) for i in range(logits.shape[0])])

    for i in range(len(logits)):
        exp = np.exp(logits[i])
        denom = np.sum(exp)

        # logit scores to probability values
        probs[i] = exp / denom

    return probs


def linearPredict(X: np.ndarray, W: np.ndarray, biases: np.ndarray):
    """
    Compute logit scores for each output class over the training data.

    X: Training data, n_samples x n_feats
    w: weight matrix, n_cls x n_feats
    biases: model biases, n_cls x 1
    n_cls: number of output classes
    """
    logits = W.dot(X.T).T + biases.squeeze(1)

    return logits


def cross_entropy(probs: np.ndarray, y: np.ndarray, W:np.ndarray) -> float:
    """
    Computes cross entropy for multiclass classification
    y: ground truth classes, n_samples x 1
    """
    n = probs.shape[0]

    L2 = np.sum(W ** 2 / (2*n))

    axis0 = np.arange(n)

    CELoss = np.log(probs[axis0, y.squeeze(-1)]).sum() + L2

    return CELoss / n


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

def plotLL(ll:np.ndarray, it:int, dataset:str, optim:str, lr:float):
    """
    Used to plot the Log likelihood curve for set iterations.
    :param ll: numpy array of Log likelihood's over training iterations
    :param it: Number of iterations
    :param dataset: Name of dataset
    :param optim: Optimization algorithm used
    :return:
    """
    x = np.arange(it)
    ax = sns.lineplot(x=x, y=ll)

    ax.set_title(f"Log-likelihood {dataset}, {optim}")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Log likelihood")
    plt.savefig(f"./plots/LL_{dataset}_{optim}_lr{lr}_i{it}.png")
    plt.show()


def main():
    """
    Loads data, normalizes data, inits weights, trains on dataset, and generates Log likelihood graphs
    """

    """
    N output classes in datasets
    """
    n_cls_digits = len(np.unique(y_digits))
    n_cls_news = len(np.unique(y_news))

    """
    N features to learn for each dataset
    """
    n_feats_digits = X_digits.shape[1]
    n_feats_news = X_news.shape[1]

    lr = .01
    iterations = 100

    """
    Weight matrices and biases for each dataset
    """
    W_digits = np.random.rand(n_cls_digits, n_feats_digits)
    biases_digits = np.random.rand(n_cls_digits, 1)

    W_news = np.random.rand(n_cls_news, n_feats_news)
    biases_news = np.random.rand(n_cls_news, 1)


    # train
    W_digits, biases_digits, LL_digits = gd(X_digits, y_digits, W_digits, biases_digits, lr, n_cls_digits, iterations)
    W_news, biases_news, LL_news = gd(X_news, y_news, W_news, biases_news, lr, n_cls_news, iterations)

    # plot log likelihood
    plotLL(LL_digits, iterations, "digits", "GD")
    plotLL(LL_news, iterations, "news", "GD")
    def predict(X_test, W, y, biases, n_cls):
        logitScores = linearPredict(X_test, W, biases)
        probs = softmax(logitScores, n_cls)

        # predicted labels
        y_hat = np.argmax(probs, axis=1).reshape(-1, 1)

        # compute accuracy
        acc = (y_hat == y).sum() / len(y) * 100

        return acc

    acc = predict(X_digits_test, W_digits, y_digits_test, biases_digits, n_cls_digits)

    print(f"Digits accuracy: {acc:.2f}%")

if  __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("-lr", )
    main()
