#%%
from array import array
import torch
import pprint
import numpy as np

#%%
data = [[1,2], [2,3]]
x_data = torch.tensor(data)
pprint.pprint(x_data)

# %%
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
pprint.pprint(x_np)

# %%
x_ones = torch.ones_like(x_data)
pprint.pprint(x_ones)

# %%
x_rand = torch.rand_like(x_data, dtype=torch.float)
pprint.pprint(x_rand)

# %%
shape = (2,3,)
rand_tensor = torch.rand(shape)
pprint.pprint(rand_tensor)
ones_tensor = torch.ones(shape)
pprint.pprint(ones_tensor)
zeros_tensor = torch.zeros(shape)
pprint.pprint(zeros_tensor)

#%%
tensor = torch.rand(3,4)
pprint.pprint(tensor.shape)
pprint.pprint(tensor.dtype)
pprint.pprint(tensor.device)

# %%
if torch.cuda.is_available():
  tensor = tensor.to('cuda')

# %%
tensor = torch.ones(4, 4)
tensor[:,1] = 0
pprint.pprint(tensor)

#%%
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# %%
pprint.pprint(tensor.T)

#%%
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
pprint.pprint(f"y1 : {y1}")
pprint.pprint(f"y2 * {y2}")
pprint.pprint(f"y3 * {y3}")

#%%
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
pprint.pprint(f"y1 : {z1}")
pprint.pprint(f"y2 * {z2}")
pprint.pprint(f"y3 * {z3}")

# %%
agg = tensor.sum()
agg_item = agg.item()  
pprint.pprint(agg_item)

# %%
tensor.add_(5)
pprint.pprint(tensor)

# %%
t = torch.ones(5)
n = t.numpy()
print(f"t : {t}")
print(f"n : {n}")

# %%
t.add_(10)
print(f"t : {t}")
print(f"n : {n}")

# %%
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t : {t}")
print(f"n : {n}")

# %%
