import torch
import numpy as np

a_np = np.array([1.0, 2.0, 3.0])
a_torch = torch.tensor([1.0, 2.0, 3.0])

print(type(a_np))
print(type(a_torch))

# Similar to numpy
print(a_torch * 2)
print(a_torch + 1)

b = torch.from_numpy(a_np)
c = a_torch.numpy()

print(type(b), type(c))

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
print(y)

y.backward()
print(x.grad)

x = torch.tensor([3.0], requires_grad=True)
y = x ** 2 + 2 * x + 1
y.backward()
print(x.grad)

print(y.grad_fn)

def manual_grad(x):
    return 2 * x + 2
print(manual_grad(3))
print(x.grad.item())

