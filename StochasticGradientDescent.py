import numpy as np
from logits import linearPredict, softmax, cross_entropy, predict, plotLL
from data import df_train_digits, df_test_digits, df_train_news, df_test_news


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
iterations = 100

"""
Weight matrices and biases for each dataset
"""
W_digits = np.random.rand(n_cls_digits, n_feats_digits)
biases_digits = np.random.rand(n_cls_digits, 1)

# W_news = np.random.rand(n_cls_news, n_feats_news)
# biases_news = np.random.rand(n_cls_news, 1)

"""
Feature and label vectors
"""
X_digits = df_train_digits.loc[:, :'X_train_65'].to_numpy()
X_digits_test = df_test_digits.loc[:, :'X_test_65'].to_numpy()
y_digits = df_train_digits.loc[:, 'Var2'].to_numpy().reshape(-1, 1)
y_digits_test = df_test_digits.loc[:, 'Var2'].to_numpy().reshape(-1, 1)

# X_news = df_train_news.loc[:, :'X_train_2001'].to_numpy()
# y_news = df_train_news.loc[:, 'Var2'].to_numpy().reshape(-1, 1)

"""
Normalize datasets
"""
X_digits = X_digits / 255.

scaler = StandardScaler()
# X_news = scaler.fit_transform(X_news)

# train
W_digits, biases_digits, LL_digits = sgd(X_digits, y_digits, W_digits, biases_digits, lr, n_cls_digits, iterations)
# W_news, biases_news, LL_news = sgd(X_news, y_news, W_news, biases_news, lr, n_cls_news, iterations)

# plot log likelihood
