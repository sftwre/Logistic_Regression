import numpy as np
from typing import List

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

