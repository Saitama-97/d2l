# -*- coding: utf-8 -*-

"""
@Time    : 2026.04.13 13:53
@File    : 4_1.py
@Project : d2l
@Author  : Nostalyn
@IDE     : PyCharm
@Desc    : 一些常用的激活函数
"""

import torch
from d2l import torch as d2l

x = torch.arange(-8, 8, 1.0, requires_grad=True)

y_relu = torch.relu(x)
y_sigmoid = torch.sigmoid(x)

d2l.plot(x.detach(), [y_relu.detach(), y_sigmoid.detach()], 'x', ['relu(x)', 'sigmoid(x)'], figsize=(5, 2.5))

d2l.plt.show()
