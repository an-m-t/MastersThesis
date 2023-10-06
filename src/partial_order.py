import general_func
import numpy as np
from data_generation import get_neighbor_vector
import random
import matplotlib.pyplot as plt
from complex_PCA import *
from vector_PCA import *

def generate_simple_tree_data(dim, num_points_per_line, tree_depth, branching_factor, bounds):
    last_end_branch = [random.uniform(min(bounds), max(bounds)) for _ in range(dim)]
    data = []
    last_data_points = []

    def create_branch(last_end_branch, data, depth):
        if depth == tree_depth:
            return data
        
        depth +=1
        for _ in range(branching_factor):
            # num_branch += 1
            direction = [random.random() for i in range(dim)]
            direction = direction / np.linalg.norm(direction)

            branch = [last_end_branch] + [last_end_branch + direction * random.uniform(min(bounds), max(bounds))for _ in range(num_points_per_line - 1)]
            data.append(branch)
            last_data_points.append(branch[-1])

        for end_points in last_data_points:
            if depth < tree_depth:
                data, _ = create_branch(end_points, data, depth)
                last_data_points.pop(0)
            else:
                return data, depth
            
        return data, depth

    data, _ = create_branch(last_end_branch, data, 1)
    direction = [-random.random() for i in range(dim)]
    direction = direction / np.linalg.norm(direction)
    first_branch = [data[0][0]] + [data[0][0] + direction * random.uniform(min(bounds), max(bounds))for _ in range(num_points_per_line - 1)]
    data = [first_branch] + data
    return data

def get_tree_vectors(data, dim):
    # Compute Vector data
    vector_data = get_neighbor_vector(data, True)
    # Remove all 0 vectors as they appeard because of separated lists
    vector_data = [p for p in vector_data if (p != [0] * dim).all()]
    return vector_data

def generate_multiple_related_points(dim, num_points_per_line, num_line, jitter):
    pass

def remove_duplicates(data):
    # Remove sequential duplicates
    flat_data = general_func.flatten(data)
    no_duplic_data = [flat_data[0]]
    for i in range(0, len(flat_data)):
        if not any((flat_data[i] == p).all() for p in no_duplic_data):
            no_duplic_data.append(flat_data[i])
    return no_duplic_data

    
def plot_reconstr_tree(dim, num_points_per_line, tree_depth, branching_factor, bounds, target_dim):
    assert(dim < 4)

    data = generate_simple_tree_data(dim, num_points_per_line, tree_depth, branching_factor, bounds)
    no_duplic_data = np.array(remove_duplicates(data))

    vector_data = np.array(get_tree_vectors(data, dim))

    transf_data = pca_transform_data(no_duplic_data, dim, target_dim, ComplexPCA, None, None)

    reconstr_data = pca_reconstruction(no_duplic_data, transf_data, target_dim, ComplexPCA, None, None)
    vector_reconstr_data, _ = vector_pca_reconstruction(no_duplic_data, vector_data, target_dim)

    fig = plt.figure()
    ax = fig.add_subplot()
    dim = str(dim) + 'd'

    try:
        ax = fig.add_subplot(111, projection=dim)
    except:
        pass

    ax.plot(*zip(*no_duplic_data), 'o', label="Original data")
    ax.plot(*zip(*reconstr_data), 'o', label="PCA")
    ax.plot(*zip(*vector_reconstr_data), 'o', label="vector PCA")

    plt.legend()

    plt.show()


def plot_transf_tree(dim, num_points_per_line, tree_depth, branching_factor, bounds, target_dim):
    assert(target_dim < 4)
    data = generate_simple_tree_data(dim, num_points_per_line, tree_depth, branching_factor, bounds)
    no_duplic_data = np.array(remove_duplicates(data))

    vector_data = np.array(get_tree_vectors(data, dim))

    transf_data = pca_transform_data(no_duplic_data, dim, target_dim, ComplexPCA, None, None)

    _, transf_vector_data = vector_pca_reconstruction(no_duplic_data, vector_data, target_dim)

    fig = plt.figure()
    ax = fig.add_subplot()
    dim = str(target_dim) + 'd'

    try:
        ax = fig.add_subplot(111, projection=dim)
    except:
        pass

    ax.plot(*zip(*transf_data), 'o', label="PCA")
    ax.plot(*zip(*transf_vector_data), 'o', label="vector PCA")

    plt.legend()

    plt.show()


