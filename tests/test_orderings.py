import sys
sys.path.insert(1, './src')
import pytest
import numpy as np
from orderings import reverse_ordering

def test_reverse_ordering():
    assert np.array_equal(reverse_ordering([1]), [1])
    assert np.array_equal(reverse_ordering([1,2]), [2,1])
    assert np.array_equal(reverse_ordering([[1,2], [3,4]]), [[3,4], [1,2]])