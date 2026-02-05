import torch

X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])  # y = 2x + 1

# Model parameters
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

def forward(X):
    return X * w + b

def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

y_pred = forward(X)
loss = mse(y_pred, y)

loss.backward()

print(w.grad, b.grad)

lr = 0.01

with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad

w.grad.zero_()
b.grad.zero_()

for epoch in range(100):
    y_pred = forward(X)
    loss = mse(y_pred, y)

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

    if epoch % 10 == 0:
        print(epoch, loss.item())
