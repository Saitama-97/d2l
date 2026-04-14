# -*- coding: utf-8 -*-

"""
@Time    : 2026.04.14 09:29
@File    : 4_2.py
@Project : d2l
@Author  : Nostalyn
@IDE     : PyCharm
@Desc    : 从零实现MLP
"""

import torch
from torch import nn
from d2l import torch as d2l
from utils.train_util import train_ch3
from matplotlib import pyplot as plt

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

loss = nn.CrossEntropyLoss(reduction='none')


def relu(X):
    """
    自定义激活函数
    Args:
        X:

    Returns:

    """
    a = torch.zeros_like(X)
    return torch.max(a, X)


def net(X):
    """
    自定义网络
    Args:
        X:

    Returns:

    """
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return H @ W2 + b2


def train():
    """
    训练
    Returns:

    """
    num_epochs, lr = 5, 0.1
    updater = torch.optim.SGD(params, lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)


def predict():
    """
    推理
    Returns:

    """
    X, y = next(iter(test_iter))
    X_show = X[:18]
    y_true = y[:18]

    with torch.no_grad():
        y_hat = net(X_show)
        y_pred = y_hat.argmax(dim=1)

    true_labels = d2l.get_fashion_mnist_labels(y_true)
    pred_labels = d2l.get_fashion_mnist_labels(y_pred)

    titles = [f'T:{t}\nP:{p}' for t, p in zip(true_labels, pred_labels)]
    d2l.show_images(X_show.reshape(18, 28, 28), 2, 9, titles=titles)
    plt.show()


if __name__ == '__main__':
    train()
    predict()
