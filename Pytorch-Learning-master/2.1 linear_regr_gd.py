import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_regression

df = pd.read_csv("lin_data.csv", index_col=0)

X = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float)
y = torch.tensor(df['y'].values, dtype=torch.float)

torch.manual_seed(123)

shuffle_idx = torch.randperm(y.size(0), dtype=torch.long)

X, y = X[shuffle_idx], y[shuffle_idx]

percent70 = int(shuffle_idx.size(0) * 0.7)

X_train, X_test = X[shuffle_idx[:percent70]], X[shuffle_idx[:percent70]]
y_train, y_test = y[shuffle_idx[:percent70]], y[shuffle_idx[percent70:]]

mu, sigma = X_train.mean(dim=0), X_train.std(dim=0)

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma


class LinearRegression1():
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(num_features, 1,
                                   dtype=torch.float)
        self.bias = torch.zeros(1, dtype=torch.float)

    def forward(self, x):
        netinputs = torch.add(torch.mm(x, self.weights), self.bias)
        activations = netinputs
        return activations.view(-1)

    def backward(self, x, yhat, y):

        grad_loss_yhat = 2*(yhat - y)

        grad_yhat_weights = x
        grad_yhat_bias = 1.

        # Chain rule: inner times outer
        grad_loss_weights = torch.mm(grad_yhat_weights.t(),
                                     grad_loss_yhat.view(-1, 1)) / y.size(0)

        grad_loss_bias = torch.sum(grad_yhat_bias*grad_loss_yhat) / y.size(0)

        # return negative gradient
        return (-1)*grad_loss_weights, (-1)*grad_loss_bias


def loss(yhat, y):
    return torch.mean((y - yhat)**2)


def train(model, x, y, num_epochs, learning_rate=0.01):
    cost = []

    for e in range(num_epochs):
        yhat = model.forward(x)

        negative_grad_w, negative_grad_b = model.backward(x, yhat, y)

        model.weights += learning_rate * negative_grad_w
        model.bias += learning_rate * negative_grad_b

        yhat = model.forward(x)
        curr_loss = loss(yhat, y)
        print("Epoch: %03d" % (e+1), end="")
        print(" | MSE: %.5f" % curr_loss)
        cost.append(curr_loss)

    return cost


model = LinearRegression1(num_features=X_train.size(1))
cost = train(model, X_train, y_train, num_epochs=200)

plt.plot(range(len(cost)), cost)
plt.show()
