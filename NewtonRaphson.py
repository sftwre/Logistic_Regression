import numpy as np
from typing import List
from logits import linearPredict, softmax, cross_entropy, predict, plotLL
from data import df_train_digits, df_test_digits, df_train_news, df_test_news

def nr(X: np.ndarray, y: np.ndarray, W: np.ndarray,
        biases: np.ndarray, lr: float, n_cls: int,
        iterations: int) -> (np.ndarray, np.ndarray, List):
    """
    Newton raphson optimization
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

        pi = np.prod(probs, axis=1)

        diagonal = np.expand_dims(pi, axis=0) * np.identity(len(pi))

        # compute error for probability of correct outcome
        probs[axis0, y.squeeze(-1)] -= 1

        # n_feats x n_feats
        hessian = X.T.dot(diagonal).dot(X)

        # first derivative w.r.t weights
        firstD = probs.T.dot(X)

        # gradient of weights and biases
        gradsW = np.linalg.inv(hessian) * firstD

        gradsBiases = np.sum(probs, axis=0).reshape(-1, 1)

        # update weights
        W -= (lr * gradsW)

        # update biases
        biases -= (lr * gradsBiases)

    return W, biases, cost_history

"""
N output classes in datasets
"""
n_cls_digits = len(np.unique(df_train_digits['Var2'].values))
n_cls_news = len(np.unique(df_train_news['Var2'].values))

"""
N features to learn for each dataset
"""
n_feats_digits = df_train_digits.shape[1] - 1
n_feats_news = df_train_news.shape[1] - 1

lr = .01
iterations = 10

"""
Weight matrices and biases for each dataset
"""
W_digits = np.random.rand(n_cls_digits, n_feats_digits)
biases_digits = np.random.rand(n_cls_digits, 1)

W_news = np.random.rand(n_cls_news, n_feats_news)
biases_news = np.random.rand(n_cls_news, 1)


# train
W_digits, biases_digits, LL_digits = nr(X_digits, y_digits, W_digits, biases_digits, lr, n_cls_digits, iterations)
# W_news, biases_news, LL_news = sgd(X_news, y_news, W_news, biases_news, lr, n_cls_news, iterations)

# plot log likelihood
plotLL(LL_digits, iterations, "digits", "SGD")
# plotLL(LL_news, iterations, "news", "SGD", lr)