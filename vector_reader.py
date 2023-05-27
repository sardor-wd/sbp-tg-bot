import numpy as np

data = np.load('vectorized_features.npz')

ids = data['ids']
vectors = data['vectors']

element = ids[0]
element2 = vectors[0]

print(f'ID: {element} ')
print(f'Vector: {element2}')