import sys
sys.path.insert(1, './src')
import numpy as np
from complex_PCA import *

# Unit tests
def test_align_phase_real_similarity():
    # Test case 1
    v1 = np.array([0.6 + 0.8j, 0.3 + 0.4j, -0.9 - 1.2j])
    u1 = np.array([0.5, 0.7, 0.9])
    v_aligned1 = rescale_eigenvectors(v1, u1)
    similarity1 = np.dot(np.real(v_aligned1), u1)
    assert similarity1 > np.dot(np.real(v1), u1)

    # Test case 2
    v2 = np.array([0.2 + 0.3j, 0.7 + 0.1j, -0.5 - 0.8j])
    u2 = np.array([-0.1, 0.9, 0.3])
    v_aligned2 = rescale_eigenvectors(v2, u2)
    similarity2 = np.dot(np.real(v_aligned2), u2) / (np.linalg.norm(np.real(v_aligned2)) * np.linalg.norm(u2))
    assert similarity2 > np.dot(np.real(v2), u2)

    # Test case 3
    v3 = np.array([0.9 + 0.1j, 0.4 - 0.6j, -0.2 + 0.8j])
    u3 = np.array([0.8, -0.2, 0.5])
    v_aligned3 = rescale_eigenvectors(v3, u3)
    similarity3 = np.dot(np.real(v_aligned3), u3) / (np.linalg.norm(np.real(v_aligned3)) * np.linalg.norm(u3))
    assert similarity3 > np.dot(np.real(v3), u3)