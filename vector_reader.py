import numpy as np

def read_vector_file(file_path):
    data = np.load(file_path)
    ids = data['ids']
    vectors = data['vectors']
    return ids, vectors

def print_ids_and_vectors(ids, vectors):
    for id, vector in zip(ids, vectors):
        print(f"ID: {id}")
        print(f"Vector: {vector}")
        print()

if __name__ == '__main__':
    file_path = 'vectorized_features.npz'  # Replace with your vector file path
    ids, vectors = read_vector_file(file_path)
    print_ids_and_vectors(ids, vectors)
