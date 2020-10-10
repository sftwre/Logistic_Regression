import numpy as np
import seaborn as sns
from data import df_train
import matplotlib.pyplot as plt

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


def linearPredict(X: np.ndarray, W: np.ndarray, biases: np.ndarray, n_cls: int):
    """
    Compute logit scores for each output class over the training data.

    X: Training data
    w: weight matrix
    biases: model biases
    n_cls: number of output classes
    """
    logits = W.dot(X.T).T

    return logits


def cross_entropy(probs: np.ndarray, y: np.ndarray, W:np.ndarray) -> float:
    """
    Computes cross entropy for multiclass classification
    y: ground truth classes, n_samples x 1
    """
    n = probs.shape[0]

    L2 = np.sum(W ** 2 / (2*n))

    axis0 = np.arange(n)

    pred_scores = probs[axis0, y.squeeze(-1)]

    outerSum = np.log(pred_scores).sum()

    CELoss = outerSum + L2

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

        logitScores = linearPredict(X, W, biases, n_cls)
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

### train
n_cls = 10
n_feats = df_train.shape[1] - 1
lr = 0.01
iterations = 1000

#  weight matrix
W = np.random.rand(n_cls, n_feats)
biases = np.random.rand(n_cls, 1)

# feature and label vectors
X = df_train.loc[:, :'X_train_65'].to_numpy()
y = df_train.loc[:, 'Var2'].to_numpy().reshape(-1, 1)

# normalize images
X = X / 255.

W, biases, costs = gd(X, y, W, biases, lr, n_cls, iterations)

# plot log likelihood as a function of the number of iterations
x = np.arange(iterations)

ax = sns.lineplot(x=x, y=costs)

ax.set_title("Log likelihood GD")
ax.set_xlabel("Iterations")
ax.set_ylabel("Log likelihood")
plt.show()
