import numpy as np
from logits import linearPredict, softmax, cross_entropy

def sgd(X: np.ndarray, y: np.ndarray, W: np.ndarray,
        biases: np.ndarray, lr: float, n_cls: int,
        iterations: int) -> (np.ndarray, np.ndarray):
    """
    Stochastic gradient descent
    """
    n = len(y)

    L2 = np.sum(1 / n * W)

    cost_history = np.zeros(iterations)

    for it in range(iterations):

        cost = 0.0

        # update weight vector based on one datapoint at a time.
        for i in range(n):
            index = np.random.randint(0, n)

            data = X[index, :].reshape(1, -1)
            label = y[index]

            logitScores = linearPredict(data, W, biases)
            probs = softmax(logitScores, n_cls)

            cost += cross_entropy(probs, label, W)

            # compute error for ground truth label
            probs[0, label] -= 1

            # gradient of weights
            gradsW = probs.T.dot(data)

            # gradient of biases
            gradsBiases = np.sum(probs, axis=0).reshape(-1, 1)

            # update weights
            W -= (lr * gradsW)

            # update biases
            biases -= (lr * gradsBiases)

        cost_history[it] = cost / n

        if it > 0 and (cost_history[it-1] > cost_history[it]):
            lr /= 10

    return W, biases, cost_history
