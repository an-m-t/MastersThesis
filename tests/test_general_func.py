import sys
sys.path.insert(1, './src')
from general_func import *
from structure_measures import *
import pytest
import numpy as np

def test_flatten():
    assert np.array_equal(flatten([]), [])
    assert np.array_equal(flatten([[], []]), [])
    assert np.array_equal(flatten([[1, 2, 3]]), [1, 2, 3])
    assert np.array_equal(flatten([[], [1], [2, 3]]), [1, 2, 3])
    assert np.array_equal(flatten([[[1, 2], [3, 4]], [[5, 6]]]), [[1, 2], [3, 4], [5, 6]])
    with pytest.raises(TypeError):
        flatten([1, 2, 3])
    with pytest.raises(ValueError):
        # design choice
        flatten([[1, [2, 3]]])


def test_measure_order_dtw():
    sequence1 = np.array([[1, 2], [3, 4], [5, 6]])
    sequence2 = np.array([[2, 4], [4, 6], [6, 8]])

    expected_distance = 2.8284271247461903
    np.allclose(measure_order_dtw(sequence1, sequence2), expected_distance)


def test_measure_order_dtw_small_changes():
    seq1 = np.array([[1,1], [3,5], [2,9], [5,9]])
    seq2 = np.array([[1,1], [3,5], [2,9], [5,8]])

    dtw = measure_order_dtw(seq1, seq2)
    dtw_orig = measure_order_dtw(seq1, seq1)

    assert dtw > dtw_orig


