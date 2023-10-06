import numpy as np
from orderings import random_ordering
import scipy
from sklearn.decomposition import PCA
import random
import complex_PCA


def rescale_eigenvectors(complex_eigenvector, orig_eigenvector):
    # Compute current similarity
    similarity = 1 - np.dot(np.real(complex_eigenvector), orig_eigenvector)

    best_similarity = similarity
    best_phase = 0.0

    for phase in np.linspace(0, 2 * np.pi, num = 100):
        complex_aligned = complex_eigenvector * np.exp(-1j * phase)
        
        similarity = 1 - np.dot(np.real(complex_aligned), orig_eigenvector)

        if similarity < best_similarity:
            best_similarity = similarity
            best_phase = phase
    
    complex_aligned = complex_eigenvector * np.exp(-1j* best_phase)
    return complex_aligned


def rescale_min_imaginary_part(vector):
    # Compute the magnitude and phase of the vector
    phase = np.angle(vector)

    # Construct a new vector with minimum imaginary part
    for i in range(len(phase)):
        aligned_vector = np.exp(-1j * np.mean(phase[i])) * vector

    mean = np.mean(vector)
    median = np.median(vector)

    # Check if the mean or median is negative
    if mean < 0 or median < 0:
        aligned_vector = -aligned_vector  # Flip the sign of all elements
    else:
        aligned_vector = -aligned_vector

    return aligned_vector


def pca_transform_data(complex_data, target_dim, own_PCA, scaling_choice, when_shuffle):
    if own_PCA != "PCA":
        pca = own_PCA(target_dim, when_shuffle, scaling_choice)
        pca.fit(complex_data)
        transformed = pca.transform()
    else:
        pca = PCA(n_components=target_dim)
        transformed = pca.fit_transform(complex_data)
    return pca, transformed


def pca_reconstruction(complex_data, target_dim, own_PCA, scaling_choice, when_shuffle):
    pca_options = {"complex_pca": complex_PCA.ComplexPCA, "complex_pca_svd_real": complex_PCA.ComplexPCA_svd_real, "complex_pca_svd_complex": complex_PCA.ComplexPCA_svd_complex, "complex_pca_svd_complex_imag_pc": complex_PCA.ComplexPCA_svd_complex_imag_pc}
    
    own_PCA = pca_options[own_PCA]
    pca, transformed = pca_transform_data(complex_data, target_dim, own_PCA, scaling_choice, when_shuffle)
    reconstruction = pca.inverse_transform(transformed)
    return reconstruction, transformed


def reconstruction_error(orig_data, reconstructed_data, real):
    if real:
        mean_list = np.mean((orig_data.real - reconstructed_data.real) * np.conj(orig_data.real - reconstructed_data.real), axis=0)**0.5
    else:
        mean_list = np.mean((orig_data - reconstructed_data) * np.conj(orig_data - reconstructed_data), axis=0)**0.5
    rsme = mean_list.sum() # using complex conjugate/ absolute value
    return rsme


def svd_flip(u, v):
    '''Flip the signs of the singular vectors to ensure consistent orientation. This prevents ambiguity in the SVD results.'''
    max_abs_cols = np.argmax(np.abs(u), axis=0) # Find the indices of the absolute values along each column 
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    v *= signs[:, np.newaxis]
    return u, v


def scale_ev_vectors(centr_matrix, ev, u, v, n_components, scaling_choice):
    if scaling_choice == "like_original_pca":
        u_orig, s_orig, vh_orig = scipy.linalg.svd(centr_matrix.real, full_matrices=True)
        compared_vectors = vh_orig[:n_components, :].conj().T
    elif scaling_choice == "like_vector_pca":
        vector_pca = ComplexPCA(n_components = n_components, when_shuffle = None, scaling_choice="like_vector_pca")
        vector_pca.fit(centr_matrix.imag)
        compared_vectors = vector_pca.components_
    elif scaling_choice == "low_imag":
        return rescale_min_imaginary_part(ev)
    elif scaling_choice == None:
        u, vh = svd_flip(u, v)
        return vh[:n_components, :].conj().T

    rescaled_vec = []
    for i in range(len(ev[0])):
        rescaled = rescale_eigenvectors(ev[:, i], compared_vectors[:, i])
        rescaled_vec.append(rescaled)
    
    return np.array(rescaled_vec).T


def complex_pca(complex_data, dimensions, own_PCA, scaling_choice = None):
    complex_pca = own_PCA(n_components=dimensions, scaling_choice=scaling_choice, when_shuffle = None)
    complex_pca.fit(complex_data)
        
    print("-------------------------------------")

    print("PCA Components: " + str(complex_pca.components_))
    print("Explained variance: " + str(complex_pca.explained_variance_))
    print("Mean: " + str(complex_pca.mean_))

    v_list = []
    for length, vector in zip(complex_pca.explained_variance_, complex_pca.components_):
        print("-> Component: " + str(vector))
        print("-> Length: " + str(np.sqrt(length)))
        v = vector * np.sqrt(length)
        v_list.append(v)

    return (complex_pca.mean_, v_list)


class ComplexPCA:
    def __init__(self, n_components, when_shuffle = None, scaling_choice = None):
        self.n_components = n_components
        self.u = self.s = self.components_ = None
        self.mean_ = None
        self.std_ = None
        self.explained_variance_ = None
        self.centr_matrix = None
        self.scaling_choice = scaling_choice
    
    @classmethod
    def whoami(self):
        return("ComplexPCA")

    def fit(self, matrix):
        n_samples, _ = matrix.shape
        self.mean_ = matrix.mean(axis=0)
        self.std_ = matrix.std()
        if self.scaling_choice in ["vector_pca", "like_vector_pca"]:
            # max_magnitude = np.max(np.linalg.norm(matrix, axis=1))
            # scaling_factor = 1 / max_magnitude
            # self.centr_matrix = scaling_factor * matrix
            self.centr_matrix = matrix - self.mean_
        elif self.scaling_choice == "id":
            self.mean_[-1] = 0
            self.centr_matrix = matrix - self.mean_
        else:
            self.centr_matrix = matrix - self.mean_
        
        # SVD on new_matrix (=X) and not cov:
        # cov = X.T @ X * 1/(n-1) symmetric 
        # -> diagonalised: cov = V @ L @ V.T 
        # where V is matrix of eig. vectors and L is diag. matrix with eig. values
        # SVD on X: 
        # X = U @ S @ V.T 
        # where U unitary, S diag. matrix of singular values
        # Altogether: 
        # cov =  X.T @ X * 1/(n-1) = V @ S @ U.T @ U @ S @ V.T * 1/(n-1) = V @ S² @ V.T * 1/ (n-1)
        u, self.s, vh = scipy.linalg.svd(self.centr_matrix, full_matrices=False)  # vh is the hermitian transpose of v
        u, vh = svd_flip(u = u, v = vh)
        self.components_ = vh[:self.n_components, :].conj().T
        explained_variance_ = (self.s ** 2) / (n_samples - 1)
        self.explained_variance_ = explained_variance_[:self.n_components]

    def transform(self):
        result = self.centr_matrix @ self.components_
        return result

    def inverse_transform(self, matrix):
        result = matrix @ self.components_.conj().T
        return result + self.mean_
    
    def get_principal_loadings(self):
        principal_loadings = [d * np.sqrt(self.explained_variance_[i]) for i,d in enumerate(self.components_)]
        return principal_loadings

class ComplexPCA_svd_real:
    def __init__(self, n_components, when_shuffle, scaling_choice = None):
        self.n_components = n_components
        self.u = self.s = self.components_ = None
        self.mean_ = None
        self.std_ = None
        self.explained_variance_ = None
        self.centr_matrix = None
        self.when_shuffle = when_shuffle
        self.shuffle_seed = 123456
    
    @classmethod
    def whoami(self):
        return("ComplexPCA_svd_real")

    def fit(self, matrix):
        if self.when_shuffle == "before_pca":
            random.seed(self.shuffle_seed)
            matrix = random_ordering(matrix)
        n_samples, _ = matrix.shape
        self.mean_ = matrix.mean(axis=0)
        self.std_ = matrix.std() 
        self.centr_matrix = matrix - self.mean_.real

        u, self.s, vh = scipy.linalg.svd(self.centr_matrix.real, full_matrices=False)  # vh is the hermitian transpose of v
        u, vh = svd_flip(u = u, v = vh)
        self.components_ = vh[:self.n_components, :].conj().T
        # for i,v in enumerate(self.components_):
        #     neg_count = len(list(filter(lambda x: (x < 0), v)))
        #     if neg_count > len(v)/2:
        #         self.components_[i] = v * -1
        explained_variance_ = (self.s ** 2) / (n_samples - 1)
        self.explained_variance_ = explained_variance_[:self.n_components]

    def transform(self):
        matrix = self.centr_matrix
        if self.when_shuffle == "before_transform":
            matrix = random_ordering(matrix)
        result = matrix @ self.components_
        return result
    # m * v = t => m * v * vt = t * vt

    def inverse_transform(self, matrix):
        result = matrix @ self.components_.T
        return result + self.mean_.real
    

class ComplexPCA_svd_complex:
    def __init__(self, n_components, when_shuffle, scaling_choice = None):
        self.n_components = n_components
        self.u = self.s = self.components_ = None
        self.mean_ = None
        self.std_ = None
        self.explained_variance_ = None
        self.centr_matrix = None
        self.when_shuffle = when_shuffle
        self.scaled_components = None
        self.scaling_choice = scaling_choice
    
    @classmethod
    def whoami(self):
        return("ComplexPCA_svd_complex")

    def fit(self, matrix):
        if self.when_shuffle == "before_pca":
            matrix = random_ordering(matrix)
        n_samples, _ = matrix.shape
        self.mean_ = matrix.mean(axis=0)
        self.std_ = matrix.std()
        self.centr_matrix = matrix - self.mean_.real
        # scipy sorts s in non decreasing order
        u, self.s, vh = scipy.linalg.svd(self.centr_matrix, full_matrices=False)  # vh is the hermitian transpose of v
        
        self.components_ = vh[:self.n_components, :].conj().T
        self.components_ = scale_ev_vectors(self.centr_matrix, self.components_, u, vh, self.n_components, self.scaling_choice)
        explained_variance_ = (self.s ** 2) / (n_samples - 1)
        self.explained_variance_ = explained_variance_[:self.n_components]

    def transform(self):
        matrix = self.centr_matrix
        if self.when_shuffle == "before_transform":
            matrix = random_ordering(matrix)
        return matrix @ np.real(self.components_)

    def inverse_transform(self, matrix):
        result = matrix @ np.real(self.components_).T
        return result + self.mean_.real
    

class ComplexPCA_svd_complex_imag_pc:
    def __init__(self, n_components, when_shuffle, scaling_choice = None):
        self.n_components = n_components
        self.u = self.s = self.components_ = None
        self.mean_ = None
        self.std_ = None
        self.explained_variance_ = None
        self.centr_matrix = None
        self.when_shuffle = when_shuffle
        self.scaled_components = None
        self.scaling_choice = scaling_choice

    def fit(self, matrix):
        if self.when_shuffle == "before_pca":
            matrix = random_ordering(matrix)
        n_samples, _ = matrix.shape
        self.mean_ = matrix.mean(axis=0)
        self.std_ = matrix.std() 
        self.centr_matrix = matrix - self.mean_.real
        # scipy sorts s in non decreasing order
        u, self.s, vh = scipy.linalg.svd(self.centr_matrix, full_matrices=True)  # vh is the hermitian transpose of v
        self.components_ = vh[:self.n_components, :].conj().T
        self.components_ = scale_ev_vectors(self.centr_matrix, self.components_, u, vh, self.n_components, self.scaling_choice)
        # self.components_ = vt[:self.n_components]
        explained_variance_ = (self.s ** 2) / (n_samples - 1)
        self.explained_variance_ = explained_variance_[:self.n_components]

    def transform(self):
        matrix = self.centr_matrix
        if self.when_shuffle == "before_transform":
            matrix = random_ordering(matrix)
        result = matrix @ self.components_.imag
        return result

    def inverse_transform(self, matrix):
        result = matrix @ self.components_.imag.T
        return result + self.mean_.imag


class ComplexPCA_svd_complex_imag_pc_on_imag:
    def __init__(self, n_components, when_shuffle, scaling_choice = None):
        self.n_components = n_components
        self.u = self.s = self.components_ = None
        self.mean_ = None
        self.std_ = None
        self.explained_variance_ = None
        self.centr_matrix = None
        self.when_shuffle = when_shuffle
        self.scaled_components = None
        self.scaling_choice = scaling_choice

    def fit(self, matrix):
        if self.when_shuffle == "before_pca":
            matrix = random_ordering(matrix)
        n_samples, _ = matrix.shape
        self.mean_ = matrix.mean(axis=0)
        self.std_ = matrix.std() 
        self.centr_matrix = matrix - self.mean_.real
        # scipy sorts s in non decreasing order
        u, self.s, vh = scipy.linalg.svd(self.centr_matrix, full_matrices=True)  # vh is the hermitian transpose of v
        self.components_ = vh[:self.n_components, :].conj().T
        # self.components_ = vt[:self.n_components]
        u_orig, s_orig, vh_orig = scipy.linalg.svd(self.centr_matrix.real, full_matrices=True)
        self.components_ = vh[:self.n_components, :].conj().T
        orig_components = vh_orig[:self.n_components, :].conj().T
        self.scaled_components = scale_ev_vectors(self.centr_matrix, self.components_, u, vh, self.n_components, self.scaling_choice)
        # self.scaled_components = np.array(self.scaled_components).T
        explained_variance_ = (self.s ** 2) / (n_samples - 1)
        self.explained_variance_ = explained_variance_[:self.n_components]

    def transform(self):
        matrix = self.centr_matrix
        if self.when_shuffle == "before_transform":
            matrix = random_ordering(matrix)
        return matrix.imag @ np.imag(self.scaled_components)
        result = matrix.imag @ self.components_.imag
        return result

    def inverse_transform(self, matrix):
        # result = matrix @ self.components_.imag.T
        result = matrix @ np.real(self.scaled_components).T
        return result + self.mean_.imag




