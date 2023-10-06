import sys
sys.path.insert(1, './src')
import pytest
import scipy
import sympy
import numpy as np
from complex_PCA import ComplexPCA, ComplexPCA_svd_complex

def check_linear_dependency(vec1, vec2, tol=1e-5):
    matrix = np.array([vec1, vec2])
    _, ind = sympy.Matrix(matrix).T.rref()
    singular_values = scipy.linalg.svd(matrix, compute_uv=False)
    numerical_rank = np.sum(singular_values > tol)
    actual_rank = np.linalg.matrix_rank(matrix)
    return numerical_rank < actual_rank


def test_complexpca_real():
    data = np.random.rand(5, 2)
    centered_data = data - np.mean(data, axis=0) 
    cov_matrix = np.cov(centered_data.T) 
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]] 
    transformed_data_manual = np.dot(centered_data, sorted_eigenvectors[:, :1])

    pca = ComplexPCA(n_components=1, when_shuffle=None)
    pca.fit(data)
    transformed_data = pca.transform()
    assert np.allclose(transformed_data - transformed_data_manual, [0]*100, rtol=1e-5, atol=1e-5) or np.allclose(transformed_data + transformed_data_manual, [0]*100, rtol=1e-5, atol=1e-5)
   
   
def test_pca_complex():
    np.random.seed(1234)

    # Test case 1: Check if Eigenvalues are same
    data = np.random.rand(100, 5) # 2 Variables and 5 observations
    centered_data = data - np.mean(data, axis=0) 
    cov_matrix = np.cov(centered_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_l = sorted(eigenvalues, key=lambda x: -x*x.conj())
    pca = ComplexPCA(n_components=5, when_shuffle=None)
    pca.fit(data)
    assert np.allclose(pca.explained_variance_, sorted_l, rtol=1e-7)

    # Test case 2: test for correct dimensions
    data = np.random.rand(100, 5)
    num_components = 3
    complexpca = ComplexPCA(num_components)
    complexpca.fit(data)
    transformed_data = complexpca.transform()
    assert transformed_data.shape == (100, num_components)

    # Test case 3: test for orthogonality
    data = np.random.rand(100, 5)  # 100 data points with 5 features each
    num_components = 5
    complexpca = ComplexPCA(num_components)
    complexpca.fit(data)
    transformed_data = complexpca.transform()
    cov_matrix = np.cov(transformed_data, rowvar=False)
    assert np.allclose(cov_matrix, np.diag(np.diagonal(cov_matrix)))


def test_pca_svd_complex():
    # Test case 1: Check if Eigenvalues are same
    data = np.random.rand(100, 5) # 2 Variables and 5 observations
    centered_data = data - np.mean(data, axis=0) 
    cov_matrix = np.cov(centered_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_l = sorted(eigenvalues, key=lambda x: -x*x.conj())
    pca = ComplexPCA_svd_complex(n_components=5, when_shuffle=None)
    pca.fit(data)
    assert np.allclose(pca.explained_variance_, sorted_l, rtol=1e-7)

    # Test case 2: test for correct dimensions
    data = np.random.rand(100, 5)
    num_components = 3
    complexpca = ComplexPCA_svd_complex(num_components, when_shuffle=None)
    complexpca.fit(data)
    transformed_data = complexpca.transform()
    assert transformed_data.shape == (100, num_components)

    # Test case 3: test for orthogonality
    data = np.random.rand(100, 5)  # 100 data points with 5 features each
    num_components = 5
    complexpca = ComplexPCA_svd_complex(num_components, when_shuffle=None)
    complexpca.fit(data)
    transformed_data = complexpca.transform()
    cov_matrix = np.cov(transformed_data, rowvar=False)
    assert np.allclose(cov_matrix, np.diag(np.diagonal(cov_matrix)))

    # Test case 4: Test if for real-valued data complexPCA and complexPCA_svd_complex return the same transformed data
    data = np.random.rand(5, 3)
    print("this test")
    num_components = 2
    complexpca_complexsvd = ComplexPCA_svd_complex(num_components, when_shuffle=None)
    complexpca_complexsvd.fit(data)
    complex_transformed_data = complexpca_complexsvd.transform()

    pca = ComplexPCA(num_components, when_shuffle=None)
    pca.fit(data)
    transformed_data = pca.transform()

    print("comp: " + str(complexpca_complexsvd.components_))
    print("orig: " + str(pca.components_))

    assert np.allclose(complexpca_complexsvd.explained_variance_, pca.explained_variance_)
    assert np.allclose(complexpca_complexsvd.components_, pca.components_)
    assert np.allclose(complex_transformed_data, transformed_data)