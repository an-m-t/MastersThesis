from scipy.spatial.distance import euclidean, mahalanobis
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix
from data_generation import *
from fastdtw import fastdtw
from complex_PCA import *
import general_func
import numpy as np
import complex_PCA
import vector_PCA
import orderings
import scipy
import logging


def get_mu_var_by_neighbor_num(data, reconstr_data, dist_measure, compare_k_neighbors):
    '''
    Calculate the expected value, variance and median of the differences between distances in the original data space and the reconstructed or in the transformed space for each nearest neighbors up to **compare_k_neighbors**.

    Parameters:
    data (array-like): Input data in original space
    reconstr_data: Input data of reconstructed data or transformed data
    dist_measure (string): either euclidean or mahalanobis
    compare_k_neighbors (integer): Compute neighbor distances until comp_k_neighbors
    alpha (bool): scaling factor for euclidean measure in the transformed space

    returns list of expected values, variances and medians of neighbors up to compare_k_neighbors
    '''
    data_num = len(data)

    if dist_measure == "euclidean":
        dist_mat = distance_matrix(data, data)
        rec_dist_mat = distance_matrix(np.real(reconstr_data), np.real(reconstr_data))
    elif dist_measure == "mahalanobis":
        dist_mat = mahalanobis_dist_matrix(data)
        rec_dist_mat = mahalanobis_dist_matrix(reconstr_data)
    elif dist_measure == "dtw":
        assert len(data[0]) == len(reconstr_data[0]), "Error: DTW works only for datasets with same dimensionality"
        result = measure_order_dtw(data, reconstr_data)
        return [result], [0], [0]
    elif dist_measure == "multiple_scalar_product":
        mean_result, var_result = scalar_product_between_multiple_neighbors(data, reconstr_data, compare_k_neighbors)
        return mean_result, var_result, [0]
    elif dist_measure == "scalar_product":
        result = scalar_product_measure(data, reconstr_data)
        return [np.mean(result)], [0], [0]

    diff_mat = np.abs(dist_mat - rec_dist_mat)

    k_exp_value = []
    k_var = []
    k_median = []

    for k in range(1, min(data_num, compare_k_neighbors) + 1):
        before_neigh = general_func.flatten([diff_mat.diagonal(-i - 1) for i in range(k)])

        after_neigh = general_func.flatten([diff_mat.diagonal(i + 1) for i in range(k)])

        neigh = np.concatenate((before_neigh, after_neigh))
        if len(neigh) != 0:
            k_exp_value.append(np.mean(neigh))
            k_median.append(np.median(neigh))
            k_var.append(np.std(neigh))

    return k_exp_value, k_var, k_median


def get_mu_var_by_neighbor_num_in_transf(original_data, transf_data, k):
    nn_orig = NearestNeighbors(n_neighbors=k + 1).fit(original_data)
    dist_orig, _ = nn_orig.kneighbors(original_data)

    nn_transf = NearestNeighbors(n_neighbors=k + 1).fit(transf_data)
    dist_transf, _ = nn_transf.kneighbors(transf_data)

    dist = np.mean(np.abs(dist_orig[:, 1:] - dist_transf[:, 1:]))
    var = np.std(np.abs(dist_orig[:, 1:] - dist_transf[:, 1:]))

    return dist, var


def measure_sep_structures(sep_points, reconstr_data, dist_measure, compare_k_neighbors):
    '''
    Separates the structures in data and applies the get_mu_var_by_neighbor_num function to measure the average error in each structure

    Parameters:
    sep_points (array-like): data to which the transformed or reconstructed data is compared to. Consists of sublists
    reconstr_data(array-like): data which is compared to sep_points
    dist_measure (string): euclidean or mahalanobis distance
    compare_k_neighbors (integer): number of neighbors up to which the average error is computed
    alpha: scaling factor for transformed space for the euclidean distance measure

    returns:
    List of the mean of all expected values, variances and medians of each sublist
    '''
    assert compare_k_neighbors < len(sep_points[0]) / 2
    if len(reconstr_data) == len(sep_points[0]) * len(sep_points):
        sep_reconstr_data = []
        line = []
        for i, p in enumerate(reconstr_data):
            if len(line) == 0:
                line.append(p)
            else:
                if p.imag.all() == 0:
                    line.append(p)
                    sep_reconstr_data.append(line)
                    line = []
                elif np.isclose([k.real + k.imag for k in line[-1]], np.real(p), atol=1e-8).all():
                    line.append(p)
    else:
        sep_reconstr_data = reconstr_data
    mus = []
    vars = []
    meds = []
    for i in range(len(sep_points)):
        mu, var, med = get_mu_var_by_neighbor_num(sep_points[i], np.real(reconstr_data[i]), dist_measure, compare_k_neighbors)
        mus.append(mu)
        vars.append(var)
        meds.append(med)

    mean_mus = [np.mean(col) for col in zip(*mus)]
    mean_vars = [np.mean(col) for col in zip(*vars)]
    mean_meds = [np.mean(col) for col in zip(*meds)]

    return mean_mus, mean_vars, mean_meds


def measure_correlation_lines(sep_line_points, transf_data, reconstr_data, num_points_per_line):
    '''
    Input:
    sep_line_points [lines containing points]: list of lists with the original data separately ordered in lines
    rec_sep_line_data [points]: sep_line_points differently ordered to compute the vectors but then transformed as sep_line_points with calculated vectors
    num_points_per_line (int): Number of points per line
    '''
    
    # Separate points to get lines
    sep_transf_sep_line_data = [transf_data[i:i + num_points_per_line] for i in range(0, len(transf_data), num_points_per_line)]
    sep_reconstr_data = [reconstr_data[i:i + num_points_per_line] for i in range(0, len(reconstr_data), num_points_per_line)]
    
    correl_coeff_mats = []
    transf_correl_coeff_mats = []
    rec_correl_coeff_mats = []
    
    # Compute correlation coefficients for each line
    for i, line in enumerate(sep_line_points):
        correl_coeff_mats.append(np.corrcoef(line, rowvar = True))
        transf_correl_coeff_mats.append(np.corrcoef(sep_transf_sep_line_data[i], rowvar = True))
        rec_correl_coeff_mats.append(np.corrcoef(sep_reconstr_data[i], rowvar = True))
    
    return correl_coeff_mats, transf_correl_coeff_mats, rec_correl_coeff_mats
    

def measure_order_kendall(d1, d2):
    return scipy.stats.kendalltau(d1,d2)


def measure_order_graphedit(d1,d2):
    len1 = len(d1)
    len2 = len(d2)

    # Create a 2D matrix to store the dynamic programming table
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Initialize the first row and column
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # Fill the dynamic programming table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if np.array_equal(d1[i - 1], d2[j - 1]):
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    # Return the edit distance
    return dp[len1][len2]


def measure_order_dtw(d1,d2):
    dtw_distance, _ = fastdtw(d1, d2, dist=euclidean)

    return dtw_distance


def measure_diff_struc(dataset1, dataset2, measure, k, sep_measure):
    pass


def mahalanobis_dist_matrix(data):
    """
    Compute the Mahalanobis distance matrix.

    Parameters:
        data (array-like): Input data matrix with shape (n_samples, n_features).

    Returns:
        distance_matrix (array-like): Mahalanobis distance matrix with shape (n_samples, n_samples).
    """
    cov = np.cov(data, rowvar = False)
    inv_cov = np.linalg.pinv(cov)
    # inv_cov = np.linalg.inv(cov)
    distance_matrix = np.zeros((len(data), len(data)))

    for i in range(len(data)):
        for j in range(len(data)):
            # distance_matrix[i, j] = mahalanobis(data[i], data[j], inv_cov)
            distance_matrix[i, j] = np.dot(np.dot((data[i] - data[j]), inv_cov), (data[i] - data[j]))
            distance_matrix[j, i] = distance_matrix[i, j]

    return distance_matrix


def compute_dyn_high_dims_vs_avg_dev(seed, target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False):
    """
    Compute average deviation vs. dimensionality of the data for dynamically increasing number of dimensions for the original space.

    Parameters:
        data (array-like): Input data matrix with shape (n_samples, n_features).
        target_dim (int): Number of dimensions specified for transformation

    Returns:
        List of means and variances for pca, vector_pca and random_projection
    """
    dim_list = general_func.generate_integer_list_dyn_high(target_dim)
    
    slurm_pickled = find_in_pickle_dyn_dims("./pickled_results/slurm/avg_dev_vs_dyn_high_dims.pickle", [seed, target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end, dim_list])
    
    pickled = find_in_pickle_dyn_dims("./pickled_results/avg_dev_vs_dyn_high_dims.pickle", [seed, target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end, dim_list])
    
    if pickled is not None:
        return pickled
    elif slurm_pickled is not None:
        logging.info("Pickle found in slurm runs")
        return slurm_pickled
    else: 
        separated = False

        list_pca_means = []
        list_pca_vars = []

        list_vector_means = []
        list_vector_vars = []

        list_rand_proj_means = []
        list_rand_proj_vars = []

        list_transf_pca_means = []
        list_transf_pca_vars = []

        list_transf_vector_means = []
        list_transf_vector_vars = []

        list_transf_random_proj_means = []
        list_transf_random_proj_vars= []

        for start_dimension in dim_list:
            # Generate data in current dimensionality
            sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dimension, jitter, bounds, parallel_start_end, False)

            # complex_data = add_complex_vector(sep_points, separated)

            if separated:
                data = flatten(sep_points)
            else:
                data = sep_points
                
            # Transform and reconstruct data with pca
            reconstr_data, transf_data = pca_reconstruction(data, target_dim, "complex_pca", None, None)

            # Transform and reconstruct data with vector pca
            vectors = get_neighbor_vector(sep_points, separated)
            reconstr_comp_data, transf_comp_data = vector_PCA.vector_pca_reconstruction(data, vectors, target_dim)
        
            l_reconstr_exp_rand_proj_values = []
            l_reconstr_rand_proj_vars = []
            l_transf_random_proj_exp_values = []
            l_transf_random_proj_vars = []

            for _ in range(1):
                transf_random_proj_data, rand_projection = general_func.random_projection(data, target_dim)
                reconstr_rand_proj_data = rand_projection.inverse_transform(transf_random_proj_data)

                if sep_measure:
                    sep_rec_rand_proj_data = [reconstr_rand_proj_data[i:i + num_points_per_line] for i in range(0, len(reconstr_rand_proj_data), num_points_per_line)]
                    rec_rand_proj_mean, rec_rand_prj_var, _ = measure_sep_structures(sep_points, sep_rec_rand_proj_data, dist_measure, k_neigh)

                    if dist_measure != "dtw":
                        sep_transf_random_proj_data = [transf_random_proj_data[i:i + num_points_per_line] for i in range(0, len(transf_random_proj_data), num_points_per_line)]
                        transf_rand_proj_mean, transf_rand_proj_var, _ = measure_sep_structures(sep_points, sep_transf_random_proj_data, dist_measure, k_neigh)
                else:
                    rec_rand_proj_mean, rec_rand_prj_var, _ = get_mu_var_by_neighbor_num(data, reconstr_rand_proj_data, dist_measure, k_neigh)
                    if dist_measure != "dtw":
                        transf_rand_proj_mean, transf_rand_proj_var, _ = get_mu_var_by_neighbor_num(data, transf_random_proj_data, dist_measure, k_neigh)
                
                l_reconstr_exp_rand_proj_values.append(rec_rand_proj_mean[-1])
                l_reconstr_rand_proj_vars.append(rec_rand_prj_var[-1])
                if dist_measure != "dtw":
                    l_transf_random_proj_exp_values.append(transf_rand_proj_mean[-1])
                    l_transf_random_proj_vars.append(transf_rand_proj_var[-1])
                else:
                    l_transf_random_proj_exp_values.append(0)
                    l_transf_random_proj_vars.append(0)
            
            reconstr_exp_rand_proj_values = np.mean(l_reconstr_exp_rand_proj_values)
            transf_random_proj_exp_values = np.mean(l_transf_random_proj_exp_values)
            reconstr_rand_proj_vars = np.mean(l_reconstr_rand_proj_vars)
            transf_random_proj_vars = np.mean(l_transf_random_proj_vars)

            # Compute exp-values and variances for reconstructed and transformed data
            if not sep_measure:
                reconstr_exp_values, reconstr_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_data, dist_measure, k_neigh)
                reconstr_exp_comp_values, reconstr_comp_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_comp_data, dist_measure, k_neigh)

                if dist_measure != "dtw":
                    transf_exp_values, transf_vars, _ = get_mu_var_by_neighbor_num(data, transf_data, dist_measure, k_neigh)
                    transf_exp_comp_values, transf_comp_vars, _ = get_mu_var_by_neighbor_num(data, transf_comp_data, dist_measure, k_neigh)

            elif sep_measure:
                sep_rec_data = [reconstr_data[i:i + num_points_per_line] for i in range(0, len(reconstr_data), num_points_per_line)]
                sep_rec_comp_data = [reconstr_comp_data[i:i + num_points_per_line] for i in range(0, len(reconstr_comp_data), num_points_per_line)]

                reconstr_exp_values, reconstr_vars, _ = measure_sep_structures(sep_points, sep_rec_data, dist_measure, k_neigh)
                reconstr_exp_comp_values, reconstr_comp_vars, _ = measure_sep_structures(sep_points, sep_rec_comp_data, dist_measure, k_neigh)

                if dist_measure != "dtw":
                    sep_transf_data = [transf_data[i:i + num_points_per_line] for i in range(0, len(transf_data), num_points_per_line)]
                    sep_transf_comp_data = [transf_comp_data[i:i + num_points_per_line] for i in range(0, len(transf_comp_data), num_points_per_line)]

                    transf_exp_values, transf_vars, _ = measure_sep_structures(sep_points, sep_transf_data, dist_measure, k_neigh)
                    transf_exp_comp_values, transf_comp_vars, _ = measure_sep_structures(sep_points, sep_transf_comp_data, dist_measure, k_neigh)

            # Append
            list_pca_means.append(reconstr_exp_values[-1])
            list_vector_means.append(reconstr_exp_comp_values[-1])
            list_rand_proj_means.append(reconstr_exp_rand_proj_values)
        
            if dist_measure != "dtw":
                list_transf_pca_vars.append(transf_vars[-1])
                list_transf_vector_vars.append(transf_comp_vars[-1])
                list_transf_random_proj_vars.append(transf_random_proj_vars)
            else:
                list_transf_pca_vars.append(0)
                list_transf_vector_vars.append(0)
                list_transf_random_proj_vars.append(0)

            if dist_measure != "dtw":
                list_transf_pca_means.append(transf_exp_values[-1])
                list_transf_vector_means.append(transf_exp_comp_values[-1])
                list_transf_random_proj_means.append(transf_random_proj_exp_values)

                list_pca_vars.append(reconstr_vars[-1])
                list_vector_vars.append(reconstr_comp_vars[-1])
                list_rand_proj_vars.append(reconstr_rand_proj_vars)
            else:
                list_transf_pca_means.append(0)
                list_transf_vector_means.append(0)
                list_transf_random_proj_means.append(0)

                list_pca_vars.append(0)
                list_vector_vars.append(0)
                list_rand_proj_vars.append(0)
                
        pickle_results_avg_dev_vs_dyn_dim("./pickled_results/avg_dev_vs_dyn_high_dims.pickle", [seed, target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end, dim_list], (list_pca_means, list_pca_vars), (list_vector_means, list_vector_vars), (list_rand_proj_means, list_rand_proj_vars),(list_transf_pca_means, list_transf_pca_vars), (list_transf_vector_means, list_transf_vector_vars), (list_transf_random_proj_means, list_transf_random_proj_vars))
        
        result = find_in_pickle_dyn_dims("./pickled_results/avg_dev_vs_dyn_high_dims.pickle", [seed, target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end, dim_list])
        
        assert result is not None
        
        return[(list_pca_means, list_pca_vars), (list_vector_means, list_vector_vars), (list_rand_proj_means, list_rand_proj_vars),(list_transf_pca_means, list_transf_pca_vars), (list_transf_vector_means, list_transf_vector_vars), (list_transf_random_proj_means, list_transf_random_proj_vars)]


def compute_dyn_low_dims_vs_avg_dev(seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False, num_runs_rand = 1):
    """
    Compute average deviation vs. dimensionality of the data for dynamically increasing number of dimensions for transformed space.

    Parameters:
        data (array-like): Input data matrix with shape (n_samples, n_features).
        start_dim (int): Number of dimensions specified for original data

    Returns:
        List of means and variances for pca, vector_pca and random_projection
    """
    logging.info("Start computing dynamic dims vs avg dev")
    
    dim_list = general_func.generate_integer_list_dyn_low(start_dim, num_lines, order)
    
    slurm_pickled = find_in_pickle_dyn_dims("./pickled_results/slurm/avg_dev_vs_dyn_low_dims.pickle", [seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end, dim_list])
    
    pickled = find_in_pickle_dyn_dims("./pickled_results/avg_dev_vs_dyn_low_dims.pickle", [seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end, dim_list])

    if pickled is not None:
        logging.info("Pickle found")
        return pickled
    elif slurm_pickled is not None:
        logging.info("Pickle found in slurm runs")
        return slurm_pickled
    else:
        logging.info("No pickle found")
        separated = False

        list_pca_means = []
        list_pca_vars = []

        list_vector_means = []
        list_vector_vars = []

        list_rand_proj_means = []
        list_rand_proj_vars = []

        list_transf_pca_means = []
        list_transf_pca_vars = []

        list_transf_vector_means = []
        list_transf_vector_vars = []

        list_transf_random_proj_means = []
        list_transf_random_proj_vars= []
        
        # Generate data in current dimensionality
        sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dim, jitter, bounds, parallel_start_end = False, standardise = False)
        if separated:
            data = flatten(sep_points)
        else:
            data = sep_points

        for target_dim in dim_list:
            logging.info("Transform and reconstruct data with pca for target_dim" + str(target_dim))
            reconstr_data, transf_data = pca_reconstruction(data, target_dim, "complex_pca", None, None)

            # Transform and reconstruct data with vector pca
            logging.info("Get vector data")
            vectors = get_neighbor_vector(sep_points, separated)
            reconstr_comp_data, transf_comp_data = vector_PCA.vector_pca_reconstruction(data, vectors, target_dim)

            # Transform and reconstruct data 10 times and get the mean
            l_reconstr_exp_rand_proj_values = []
            l_reconstr_rand_proj_vars = []
            l_transf_random_proj_exp_values = []
            l_transf_random_proj_vars = []
            
            logging.info("Compute random projections")
            
            for r in range(num_runs_rand):
                logging.info("Random projections round " + str(r))
                transf_random_proj_data, rand_projection = general_func.random_projection(data, target_dim)
                reconstr_rand_proj_data = rand_projection.inverse_transform(transf_random_proj_data)

                if sep_measure:
                    sep_rec_rand_proj_data = [reconstr_rand_proj_data[i:i + num_points_per_line] for i in range(0, len(reconstr_rand_proj_data), num_points_per_line)]
                    rec_rand_proj_mean, rec_rand_prj_var, _ = measure_sep_structures(sep_points, sep_rec_rand_proj_data, dist_measure, k_neigh)
                    
                    if dist_measure != "dtw":
                        sep_transf_random_proj_data = [transf_random_proj_data[i:i + num_points_per_line] for i in range(0, len(transf_random_proj_data), num_points_per_line)]
                        transf_rand_proj_mean, transf_rand_proj_var, _ = measure_sep_structures(sep_points, sep_transf_random_proj_data, dist_measure, k_neigh)
                else:
                    rec_rand_proj_mean, rec_rand_prj_var, _ = get_mu_var_by_neighbor_num(data, reconstr_rand_proj_data, dist_measure, k_neigh)
                    if dist_measure != "dtw":
                        transf_rand_proj_mean, transf_rand_proj_var, _ = get_mu_var_by_neighbor_num(data, transf_random_proj_data, dist_measure, k_neigh)
                
                l_reconstr_exp_rand_proj_values.append(rec_rand_proj_mean[-1])
                l_reconstr_rand_proj_vars.append(rec_rand_prj_var[-1])
                if dist_measure != "dtw":
                    l_transf_random_proj_exp_values.append(transf_rand_proj_mean[-1])
                    l_transf_random_proj_vars.append(transf_rand_proj_var[-1])
                else:
                    l_transf_random_proj_exp_values.append(0)
                    l_transf_random_proj_vars.append(0)
            reconstr_exp_rand_proj_values = np.mean(l_reconstr_exp_rand_proj_values)
            transf_random_proj_exp_values = np.mean(l_transf_random_proj_exp_values)
            reconstr_rand_proj_vars = np.mean(l_reconstr_rand_proj_vars)
            transf_random_proj_vars = np.mean(l_transf_random_proj_vars)

            # Compute exp-values and variances for reconstructed and transformed data
            logging.info("Compute expected values, variances and median")
            if not sep_measure:
                reconstr_exp_values, reconstr_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_data, dist_measure, k_neigh)
                reconstr_exp_comp_values, reconstr_comp_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_comp_data, dist_measure, k_neigh)

                if dist_measure != "dtw":
                    transf_exp_values, transf_vars, _ = get_mu_var_by_neighbor_num(data, transf_data, dist_measure, k_neigh)
                    transf_exp_comp_values, transf_comp_vars, _ = get_mu_var_by_neighbor_num(data, transf_comp_data, dist_measure, k_neigh)

            elif sep_measure:
                sep_rec_data = [reconstr_data[i:i + num_points_per_line] for i in range(0, len(reconstr_data), num_points_per_line)]
                sep_rec_comp_data = [reconstr_comp_data[i:i + num_points_per_line] for i in range(0, len(reconstr_comp_data), num_points_per_line)]

                reconstr_exp_values, reconstr_vars, _ = measure_sep_structures(sep_points, sep_rec_data, dist_measure, k_neigh)
                reconstr_exp_comp_values, reconstr_comp_vars, _ = measure_sep_structures(sep_points, sep_rec_comp_data, dist_measure, k_neigh)

                if dist_measure != "dtw":
                    sep_transf_data = [transf_data[i:i + num_points_per_line] for i in range(0, len(transf_data), num_points_per_line)]
                    sep_transf_comp_data = [transf_comp_data[i:i + num_points_per_line] for i in range(0, len(transf_comp_data), num_points_per_line)]

                    transf_exp_values, transf_vars, _ = measure_sep_structures(sep_points, sep_transf_data, dist_measure, k_neigh)
                    transf_exp_comp_values, transf_comp_vars, _ = measure_sep_structures(sep_points, sep_transf_comp_data, dist_measure, k_neigh)

            # Append
            list_pca_means.append(reconstr_exp_values[-1])
            list_vector_means.append(reconstr_exp_comp_values[-1])
            list_rand_proj_means.append(reconstr_exp_rand_proj_values)

            list_pca_vars.append(reconstr_vars[-1])
            list_vector_vars.append(reconstr_comp_vars[-1])
            list_rand_proj_vars.append(reconstr_rand_proj_vars)

            if dist_measure != "dtw":
                list_transf_pca_means.append(transf_exp_values[-1])
                list_transf_vector_means.append(transf_exp_comp_values[-1])
                list_transf_random_proj_means.append(transf_random_proj_exp_values)

                list_transf_pca_vars.append(transf_vars[-1])
                list_transf_vector_vars.append(transf_comp_vars[-1])
                list_transf_random_proj_vars.append(transf_random_proj_vars)
            else:
                list_transf_pca_means.append(0)
                list_transf_vector_means.append(0)
                list_transf_random_proj_means.append(0)

                list_transf_pca_vars.append(0)
                list_transf_vector_vars.append(0)
                list_transf_random_proj_vars.append(0)
            
        logging.info("Finished computing")
        
        pickle_results_avg_dev_vs_dyn_dim("./pickled_results/avg_dev_vs_dyn_low_dims.pickle", [seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end, dim_list], (list_pca_means, list_pca_vars), (list_vector_means, list_vector_vars), (list_rand_proj_means, list_rand_proj_vars),(list_transf_pca_means, list_transf_pca_vars), (list_transf_vector_means, list_transf_vector_vars), (list_transf_random_proj_means, list_transf_random_proj_vars))
        
        result = find_in_pickle_dyn_dims("./pickled_results/avg_dev_vs_dyn_low_dims.pickle", [seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end, dim_list])
        
        assert result is not None
        
        return[(list_pca_means, list_pca_vars), (list_vector_means, list_vector_vars), (list_rand_proj_means, list_rand_proj_vars),(list_transf_pca_means, list_transf_pca_vars), (list_transf_vector_means, list_transf_vector_vars), (list_transf_random_proj_means, list_transf_random_proj_vars)]


def scalar_product_measure(flat_original_data, transformed_data):
    assert len(flat_original_data) == len(transformed_data)

    zigzag_vectors = get_neighbor_vector(flat_original_data, False)
    transf_zigzag_vectors = get_neighbor_vector(transformed_data, False)

    # Compute scalar products
    scal_prods_list = []
    transf_scal_prod_list = []

    for i in range(len(zigzag_vectors) - 1):
        scal_prod = np.dot(zigzag_vectors[i], zigzag_vectors[i + 1])

        transf_scal_prod = np.dot(transf_zigzag_vectors[i], transf_zigzag_vectors[i + 1])
        scal_prods_list.append(scal_prod)
        transf_scal_prod_list.append(transf_scal_prod)

    result_list = np.abs(np.subtract(scal_prods_list, transf_scal_prod_list))
    return result_list


def scalar_product_between_multiple_neighbors(data1, data2, k):
    n = len(data1)
    mean_results = []
    var_results = []
    all_scalars = []

    for j in range(1, k + 1):
        scalars = []
        for i in range(n):
            # Calculate indices of neighbors
            prev_idx = i - j
            next_idx = i + j

            # Check if indices are within bounds
            if prev_idx >= 0 and next_idx < n:
                vec11 = data1[i] - data1[prev_idx]
                vec12 = data1[next_idx] - data1[i]
                
                vec21 = data2[i] - data2[prev_idx]
                vec22 = data2[next_idx] - data2[i]
                
                # Compute scalar product
                scalar_product_1 = np.dot(vec11, vec12)
                scalar_product_2 = np.dot(vec21, vec22)
                scalars.append(np.abs(scalar_product_1 - scalar_product_2))
            
        if len(scalars) != 0:
            if len(all_scalars) != 0:
                scalars.extend(all_scalars[-1])
                all_scalars.append(scalars)
            else:
                all_scalars.append(scalars)
            mean_results.append(np.mean(scalars))
            var_results.append(np.var(scalars))
    return mean_results, var_results
