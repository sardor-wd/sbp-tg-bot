import numpy as np

data = np.load('vectorized_features.npz')

ids = data['ids']
vectors = data['vectors']

element = ids[20]
element2 = vectors[20]

print(element)
print(element2)