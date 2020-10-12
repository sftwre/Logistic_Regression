import numpy as np
import seaborn as sns
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

def plotLL(ll:np.ndarray, it:int, dataset:str, optim:str, lr:float, acc:float):
    """
    Used to plot the Log likelihood curve for set iterations.
    :param ll: numpy array of Log likelihood's over training iterations
    :param it: Number of iterations
    :param dataset: Name of dataset
    :param optim: Optimization algorithm used
    :param lr: learning rate
    :param acc: Test accuracy
    :return:
    """
    x = np.arange(it)
    ax = sns.lineplot(x=x, y=ll)

    ax.set_title(f"Log-likelihood {dataset}, {optim}, {acc:.2f}% accuracy")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Log likelihood")
    plt.savefig(f"./plots/LL_{dataset}_{optim}_lr{lr}_i{it}.png")
    plt.show()

def predict(X:np.ndarray, y:np.ndarray, W:np.ndarray, biases:np.ndarray, n_cls:int):
    """
    Computes accuracy or test error on test set.
    :param X: Input data
    :param y: ground truth labels
    :param W: Learned weights
    :param biases:
    :param n_cls: number of output classes
    """
    logitScores = linearPredict(X, W, biases)
    probs = softmax(logitScores, n_cls)

    # predicted labels
    y_hat = np.argmax(probs, axis=1).reshape(-1, 1)

    # compute accuracy
    acc = (y_hat == y).sum() / len(y) * 100

    return acc