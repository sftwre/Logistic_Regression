import numpy as np
from GradientDescent import linearPredict, softmax, cross_entropy

def sgd(X: np.ndarray, y: np.ndarray, W: np.ndarray,
        biases: np.ndarray, lr: float, n_cls: int,
        iterations: int) -> (np.ndarray, np.ndarray):
    """
    Stochastic gradient descent
    """
    n = len(y)

    L1 = 2 / n * W

    axis0 = np.arange(X.shape[0])

    cost_history = np.zeros(iterations)

    for it in range(iterations):

        cost = 0.0

        # update weight vector based on one datapoint at a time.

        for i in range(n):
            index = np.random.randint(0, n)

            logitScores = linearPredict(X[index, :], W, biases, n_cls)
            probs = softmax(logitScores, n_cls)

            cost += cross_entropy(probs, y[index])

            # gradient of weights
            gradsW = (probs[axis0, y.squeeze(-1)] - 1).T.dot(X)
            gradsBiases = np.sum(probs, axis=0).reshape(-1, 1)

            # update weights
            W -= (lr * gradsW)

            # update biases
            biases -= (lr * gradsBiases)

        cost_history[it] = cost

    return W, biases, cost_history