import sys
sys.path.insert(1, './src')
import pytest
import numpy as np
from structure_measures import *
from unittest import TestCase

def test_structure_measure_euclidean():
    # Testcase 1:
    data = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
    reconstr_data = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
    dist_measure = "euclidean"
    compare_k_neighbors = 3
    alpha = False

    result = get_mu_var_by_neighbor_num(data, reconstr_data, dist_measure, compare_k_neighbors)

    assert len(result[0]) == compare_k_neighbors
    assert len(result[1]) == compare_k_neighbors
    assert len(result[2]) == compare_k_neighbors
    assert np.array_equal(result[0], [0.0] * compare_k_neighbors)
    assert np.array_equal(result[1], [0.0] * compare_k_neighbors)
    assert np.array_equal(result[2], [0.0] * compare_k_neighbors)

    # Testcase 2: 
    data = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
    reconstr_data = [[0, 0]]*5
    dist_measure = "euclidean"
    compare_k_neighbors = 3
    alpha = False

    result = get_mu_var_by_neighbor_num(data, reconstr_data, dist_measure, compare_k_neighbors)

    assert len(result[0]) == compare_k_neighbors
    assert len(result[1]) == compare_k_neighbors
    assert len(result[2]) == compare_k_neighbors

def test_scalar_products_functions():
    
    data1 = list(range(8))
    data2 = [i % 3 for i in data1]
    
    k = 1
    
    # mult_result = scalar_product_between_multiple_neighbors(data1, data2, k)
    # one_result = scalar_product_measure(data1, data2)
    # assert np.array_equal(mult_result, [np.mean(one_result)])
    
    zigzag, separated = generate_correct_ordered_data("zigzag", 100, 2, 100, 10, [0, 100])
    one_line, _ = generate_correct_ordered_data("one_line", 100, 2, 100, 10, [0, 100])
    data1 = list(range(5))
    data2 = [i % 3 for i in data1]
    
    k = 1
    
    mult_result = scalar_product_between_multiple_neighbors(zigzag, one_line, k)
    assert np.mean(scalar_product_measure(zigzag, one_line)) == mult_result[0][0]