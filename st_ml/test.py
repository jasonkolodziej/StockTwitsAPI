import numpy as np

test = [
    [0,1,1],
    [0,1,1],
    [0,1,1],
    [0,1,1]
]

test_np = np.random.rand(1,200)

print(test_np.shape)

test_np.resize((10,20))

print(test_np.shape)
print(test_np)