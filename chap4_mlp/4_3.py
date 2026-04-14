# -*- coding: utf-8 -*-

"""
@Time    : 2026.04.14 14:19
@File    : 4_3.py
@Project : d2l
@Author  : Nostalyn
@IDE     : PyCharm
@Desc    : 多层感知机的简洁实现
"""

import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt

from utils.train_util import train_ch3

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def predict(test_iter, n=18):
    X, y = next(iter(test_iter))
    X_show = X[:n]
    y_true = y[:n]

    net.eval()

    with torch.no_grad():
        y_hat = net(X_show)
        y_pred = y_hat.argmax(dim=1)

    true_labels = d2l.get_fashion_mnist_labels(y_true)
    pred_labels = d2l.get_fashion_mnist_labels(y_pred)

    titles = [f'T:{t}\nP:{p}' for t, p in zip(true_labels, pred_labels)]
    d2l.show_images(X_show.squeeze(1), 2, n // 2, titles=titles)
    plt.show()


def main():
    net.apply(init_weights)

    batch_size, lr, num_epochs = 256, 0.1, 10

    loss = nn.CrossEntropyLoss(reduction='none')

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer)

    predict(test_iter)


if __name__ == '__main__':
    main()
