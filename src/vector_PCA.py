import scipy
import numpy as np

def vector_pca_transform_data(original_data, vector_data, target_dim):
    pca = VectorPCA(target_dim)
    pca.fit(original_data, vector_data)
    transformed = pca.transform()
    return pca, transformed


def vector_pca_reconstruction(original_data, vector_data, target_dim):
    pca, transf_data = vector_pca_transform_data(original_data, vector_data, target_dim)
    reconst_data = pca.inverse_transform(transf_data)
    return reconst_data, transf_data 


def svd_flip(u, v):
    '''Flip the signs of the singular vectors to ensure consistent orientation. This prevents ambiguity in the SVD results.'''
    max_abs_cols = np.argmax(np.abs(u), axis=0) # Find the indices of the absolute values along each column 
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    v *= signs[:, np.newaxis]
    return u, v


class VectorPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.centr_vector_data = None
        self.data = None
        self.centr_data = None
        self.data_mean = None
    
    @classmethod
    def whoami(self):
        return("VectorPCA")

    def fit(self, matrix, vector_data):
        n_samples, _ = matrix.shape
        self.mean_ = vector_data.mean(axis=0)
        self.centr_vector_data = vector_data - self.mean_
        self.data = matrix
        self.data_mean = matrix.mean(axis = 0)
        self.centr_data = matrix - self.data_mean
        
        u, self.s, vh = scipy.linalg.svd(self.centr_vector_data, full_matrices=False)  # vh is the hermitian transpose of v
        u, vh = svd_flip(u = u, v = vh)
        self.components_ = vh[:self.n_components, :].conj().T
        explained_variance_ = (self.s ** 2) / (n_samples - 1)
        self.explained_variance_ = explained_variance_[:self.n_components]

    def transform(self):
        result = self.centr_data @ self.components_
        return result

    def inverse_transform(self, transf_matrix):
        result = transf_matrix @ self.components_.conj().T
        return result + self.data_mean
    
    def get_principal_loadings(self):
        principal_loadings = [d * np.sqrt(self.explained_variance_[i]) for i,d in enumerate(self.components_)]
        return principal_loadings