from sklearn.decomposition import PCA
import numpy as np

def pca(points, dimensions):
    pca = PCA(n_components=dimensions)
    pca.fit(points)
    print("-------------------------------------")
    print("Real PCA Components: " + str(pca.components_))
    print("Real explained variance: " + str(pca.explained_variance_))
    print("Real mean: " + str(pca.mean_))
    v_list = []
    for length, vector in zip(pca.explained_variance_, pca.components_):
        print("-> Component: " + str(vector))
        print("-> Length: " + str(np.sqrt(length)))
        v = vector * np.sqrt(length)
        v_list.append(v)

    return (pca.mean_, v_list)


# Quick own implementation of pca to test sklearn implementation
def pca_test(flat_points, dimensions):
    X_meaned = flat_points - np.mean(flat_points, axis = 0)    
    cov_mat = np.cov(X_meaned, rowvar=False)
    x, y = zip(*flat_points)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalue = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:,sorted_index]
    list = pca(flat_points, dimensions)
    