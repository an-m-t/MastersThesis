import numpy as np
from sklearn.decomposition import FastICA
import orderings
from general_func import *
from data_generation import *
import matplotlib.pyplot as plt
from structure_measures import *
from deeptime.decomposition import TICA
import logging


def compute_ica_dyn_high_dims_vs_avg_dev(target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False):
    """
    Compute average deviation vs. dimensionality of the data for dynamically increasing number of dimensions for the original space.

    Parameters:
        data (array-like): Input data matrix with shape (n_samples, n_features).
        target_dim (int): Number of dimensions specified for transformation

    Returns:
        List of means and variances for pca, vector_pca and random_projection
    """

    dim_list = general_func.generate_integer_list_dyn_high(target_dim)
    separated = False

    list_pca_means = []
    list_pca_vars = []

    list_vector_means = []
    list_vector_vars = []

    list_transf_pca_means = []
    list_transf_pca_vars = []

    list_transf_vector_means = []
    list_transf_vector_vars = []
    
    list_transf_tica_means = []
    list_transf_tica_vars = []

    for start_dimension in dim_list:
        # Generate data in current dimensionality
        sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dimension, jitter, bounds, parallel_start_end)

        if separated:
            data = flatten(sep_points)
        else:
            data = sep_points

        vector_data = get_neighbor_vector(sep_points, separated)
        
        # Transform data
        ica = FastICA(n_components=target_dim)
        ica_transf = ica.fit_transform(data)
        ica_reconstr = ica.inverse_transform(ica_transf)

        vector_ica = FastICA(n_components=target_dim, whiten="unit-variance")
        vector_ica.fit(vector_data)
        vector_ica_transf = vector_ica.transform(data)
        vector_ica_reconstr = vector_ica.inverse_transform(vector_ica_transf)
        
        tica = TICA(dim = target_dim, lagtime = 1)
        tica_transformed = tica.fit_transform(data)

        # Compute exp-values and variances for reconstructed and transformed data
        if not sep_measure:
            reconstr_exp_values, reconstr_vars, _ = get_mu_var_by_neighbor_num(data, ica_reconstr, dist_measure, k_neigh)
            reconstr_exp_comp_values, reconstr_comp_vars, _ = get_mu_var_by_neighbor_num(data, vector_ica_reconstr, dist_measure, k_neigh)

            transf_exp_values, transf_vars, _ = get_mu_var_by_neighbor_num(data, ica_transf, dist_measure, k_neigh)
            transf_exp_comp_values, transf_comp_vars, _ = get_mu_var_by_neighbor_num(data, vector_ica_transf, dist_measure, k_neigh)
            transf_exp_tica_values, transf_tica_vars, _ = get_mu_var_by_neighbor_num(data, tica_transformed, dist_measure, k_neigh)
             
        elif sep_measure:
            sep_rec_data = [ica_reconstr[i:i + num_points_per_line] for i in range(0, len(ica_reconstr), num_points_per_line)]
            sep_rec_comp_data = [vector_ica_reconstr[i:i + num_points_per_line] for i in range(0, len(vector_ica_reconstr), num_points_per_line)]

            reconstr_exp_values, reconstr_vars, _ = measure_sep_structures(sep_points, sep_rec_data, dist_measure, k_neigh)
            reconstr_exp_comp_values, reconstr_comp_vars, _ = measure_sep_structures(sep_points, sep_rec_comp_data, dist_measure, k_neigh)

            sep_transf_data = [ica_transf[i:i + num_points_per_line] for i in range(0, len(ica_transf), num_points_per_line)]
            sep_transf_comp_data = [vector_ica_transf[i:i + num_points_per_line] for i in range(0, len(vector_ica_transf), num_points_per_line)]
            sep_transf_tica_data = [tica_transformed[i:i + num_points_per_line] for i in range(0, len(tica_transformed), num_points_per_line)]

            transf_exp_values, transf_vars, _ = measure_sep_structures(sep_points, sep_transf_data, dist_measure, k_neigh)
            transf_exp_comp_values, transf_comp_vars, _ = measure_sep_structures(sep_points, sep_transf_comp_data, dist_measure, k_neigh)
            transf_exp_tica_values, transf_tica_vars, _ = measure_sep_structures(sep_points, sep_transf_tica_data, dist_measure, k_neigh)

        # Append
        list_pca_means.append(reconstr_exp_values[-1])
        list_vector_means.append(reconstr_exp_comp_values[-1])

        list_transf_pca_means.append(transf_exp_values[-1])
        list_transf_vector_means.append(transf_exp_comp_values[-1])

        list_pca_vars.append(reconstr_vars[-1])
        list_vector_vars.append(reconstr_comp_vars[-1])

        list_transf_pca_vars.append(transf_vars[-1])
        list_transf_vector_vars.append(transf_comp_vars[-1])
        
        list_transf_tica_means.append(transf_exp_tica_values[-1])
        list_transf_tica_vars.append(transf_tica_vars[-1])

    return[(list_pca_means, list_pca_vars), (list_vector_means, list_vector_vars),(list_transf_pca_means, list_transf_pca_vars), (list_transf_vector_means, list_transf_vector_vars), (list_transf_tica_means, list_transf_tica_vars)]


def compute_ica_dyn_low_dims_vs_avg_dev(seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False):
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
    separated = False
    
    slurm_pickled = find_in_ica_pickle_dyn_dims("./pickled_results/slurm/ica_avg_dev_vs_dyn_low_dims.pickle", [seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end, dim_list])
    
    pickled = find_in_ica_pickle_dyn_dims("./pickled_results/ica_avg_dev_vs_dyn_low_dims.pickle", [seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end, dim_list])
    
    if pickled is not None:
        logging.info("Pickle found")
        return pickled
    elif slurm_pickled is not None:
        logging.info("Pickle found in slurm runs")
        return slurm_pickled
    else:
        logging.info("No pickle found")
        list_ica_means = []
        list_ica_vars = []

        list_vector_ica_means = []
        list_vector_ica_vars = []

        list_transf_ica_means = []
        list_transf_ica_vars = []

        list_transf_vector_ica_means = []
        list_transf_vector_ica_vars = []

        list_transf_tica_means = []
        list_transf_tica_vars= []

        # Generate data in current dimensionality
        sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dim, jitter, bounds, parallel_start_end)

        if separated:
            data = flatten(sep_points)
        else:
            data = sep_points
            
        vector_data = get_neighbor_vector(sep_points, separated)

        for target_dim in dim_list:
            logging.info("Transform and reconstruct data with pca for target_dim" + str(target_dim))
            ica = FastICA(n_components=target_dim)
            ica_transf = ica.fit_transform(data)
            ica_reconstr = ica.inverse_transform(ica_transf)

            vector_ica = FastICA(n_components=target_dim, whiten = "unit-variance")
            vector_ica.fit(vector_data)
            vector_ica_transf = vector_ica.transform(data)
            vector_ica_reconstr = vector_ica.inverse_transform(vector_ica_transf)
            
            tica = TICA(dim = target_dim, lagtime = 1)
            tica_transf = tica.fit_transform(data)

            # Compute exp-values and variances for reconstructed and transformed data
            if not sep_measure:
                reconstr_exp_values, reconstr_vars, _ = get_mu_var_by_neighbor_num(data, ica_reconstr, dist_measure, k_neigh)
                reconstr_exp_comp_values, reconstr_comp_vars, _ = get_mu_var_by_neighbor_num(data, vector_ica_reconstr, dist_measure, k_neigh)

                transf_exp_values, transf_vars, _ = get_mu_var_by_neighbor_num(data, ica_transf, dist_measure, k_neigh)
                transf_exp_comp_values, transf_comp_vars, _ = get_mu_var_by_neighbor_num(data, vector_ica_transf, dist_measure, k_neigh)
                transf_exp_tica_values, transf_tica_vars, _ = get_mu_var_by_neighbor_num(data, tica_transf, dist_measure, k_neigh)

            elif sep_measure:
                sep_rec_data = [ica_reconstr[i:i + num_points_per_line] for i in range(0, len(ica_reconstr), num_points_per_line)]
                sep_rec_comp_data = [vector_ica_reconstr[i:i + num_points_per_line] for i in range(0, len(vector_ica_reconstr), num_points_per_line)]

                reconstr_exp_values, reconstr_vars, _ = measure_sep_structures(sep_points, sep_rec_data, dist_measure, k_neigh)
                reconstr_exp_comp_values, reconstr_comp_vars, _ = measure_sep_structures(sep_points, sep_rec_comp_data, dist_measure, k_neigh)

                sep_transf_data = [ica_transf[i:i + num_points_per_line] for i in range(0, len(ica_transf), num_points_per_line)]
                sep_transf_comp_data = [vector_ica_transf[i:i + num_points_per_line] for i in range(0, len(vector_ica_transf), num_points_per_line)]
                sep_transf_tica_data = [tica_transf[i:i + num_points_per_line] for i in range(0, len(tica_transf), num_points_per_line)]

                transf_exp_values, transf_vars, _ = measure_sep_structures(sep_points, sep_transf_data, dist_measure, k_neigh)
                transf_exp_comp_values, transf_comp_vars, _ = measure_sep_structures(sep_points, sep_transf_comp_data, dist_measure, k_neigh)
                transf_exp_tica_values, transf_tica_vars, _ = measure_sep_structures(sep_points, sep_transf_tica_data, dist_measure, k_neigh)

            # Append
            list_ica_means.append(reconstr_exp_values[-1])
            list_vector_ica_means.append(reconstr_exp_comp_values[-1])

            list_transf_ica_means.append(transf_exp_values[-1])
            list_transf_vector_ica_means.append(transf_exp_comp_values[-1])

            list_ica_vars.append(reconstr_vars[-1])
            list_vector_ica_vars.append(reconstr_comp_vars[-1])

            list_transf_ica_vars.append(transf_vars[-1])
            list_transf_vector_ica_vars.append(transf_comp_vars[-1])

            list_transf_tica_means.append(transf_exp_tica_values[-1])
            list_transf_tica_vars.append(transf_tica_vars[-1])
            
        pickle_ica_results_avg_dev_vs_dyn_dim("./pickled_results/ica_avg_dev_vs_dyn_low_dims.pickle", [seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end, dim_list], (list_ica_means, list_ica_vars), (list_vector_ica_means, list_vector_ica_vars),(list_transf_ica_means, list_transf_ica_vars), (list_transf_vector_ica_means, list_transf_vector_ica_vars), (list_transf_tica_means, list_transf_tica_vars))
            
        return(list_ica_means, list_ica_vars), (list_vector_ica_means, list_vector_ica_vars), (list_transf_ica_means, list_transf_ica_vars), (list_transf_vector_ica_means, list_transf_vector_ica_vars), (list_transf_tica_means, list_transf_tica_vars)


def plot_ica_reconstr_dyn_high_dims_vs_avg_dev(target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False):
    '''
    Plot the average deviation vs. the dynamic changing higher dimension in the original/ reconstructed space. x-axis: number of dimensions which are transformed to target_dim and the reconstructed.

    Parameters:
    target_dim (integer): specifies to what dimension data is to be transformed
    order (string): data is generated for every higher dimension in specified order
    bounds (tuple(integer)): data is generated for every higher dimension in specified bounds
    num_lines (integer): number of lines which are generated for specified order
    num_points_per_line (integer): number of points on each line
    dist_measure (string): euclidean or mahalanobis distance measure
    k_neigh (integer): number how many neighbors should be compared
    sep_measure (string): specifies if lines are supposed to measured separately. Only for orders suitable for this!
    jitter (integer): how far the points are laying from the lines

    Returns: 
    nothing (plot)
    '''
    dim_list = generate_integer_list_dyn_high(target_dim)
    
    pca_mean_var, vector_mean_var, _, _ , _ = compute_ica_dyn_high_dims_vs_avg_dev(target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end)


    # Plot for ICA for reconstructed data
    upper_var = np.array(pca_mean_var[0]) + np.array(pca_mean_var[1])
    lower_var = np.array(pca_mean_var[0]) - np.array(pca_mean_var[1])

    if sep_measure:
        plt.plot(dim_list, pca_mean_var[0], label="Expected Value: ICA", color = "c")
        plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.3, color = "c")
    else: 
        plt.plot(dim_list, pca_mean_var[0], label="Expected Value: ICA", color = "c")
        plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.6, color = "c", linewidth = 2)

    # Plot for Vector ICA for reconstructed data
    upper_var = np.array(vector_mean_var[0]) + np.array(vector_mean_var[1])
    lower_var = np.array(vector_mean_var[0]) - np.array(vector_mean_var[1])
    
    if sep_measure:
        plt.plot(dim_list, vector_mean_var[0], label="Expected Value: Vector ICA", color = "orange")
        plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.3, color = "orange")
    else:
        plt.plot(dim_list, vector_mean_var[0], label="Expected Value: Vector ICA", color = "orange")
        plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.6, color = "orange", linewidth = 2)

    plt.xlabel("Number of dimensions reduced to " + str(target_dim) + " and reconstructed")
    plt.ylabel("Average deviation from original data")
    plt.title("Reconstructed: Average devation vs. dynamic dimension for original space")
    plt.legend()
    plt.show()   


def plot_ica_transf_dyn_high_dims_vs_avg_dev(target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False):
    '''
    Plot the average deviation vs. the dynamic changing higher dimension in the original/ transformed space. x-axis: number of dimensions which are transformed to target_dim.

    Parameters:
    target_dim (integer): specifies to what dimension data is to be transformed
    order (string): data is generated for every higher dimension in specified order
    bounds (tuple(integer)): data is generated for every higher dimension in specified bounds
    num_lines (integer): number of lines which are generated for specified order
    num_points_per_line (integer): number of points on each line
    dist_measure (string): euclidean or mahalanobis distance measure
    k_neigh (integer): number how many neighbors should be compared
    sep_measure (string): specifies if lines are supposed to measured separately. Only for orders suitable for this!
    jitter (integer): how far the points are laying from the lines

    Returns: 
    nothing (plot)
    '''
    _, _, transf_pca_mean_var, transf_vector_mean_var, transf_tica_mean_var = compute_ica_dyn_high_dims_vs_avg_dev(target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end)

    dims_list = generate_integer_list_dyn_high(target_dim)

    # Plot for ICA for reconstructed data
    upper_var = np.array(transf_pca_mean_var[0]) + np.array(transf_pca_mean_var[1])
    lower_var = np.array(transf_pca_mean_var[0]) - np.array(transf_pca_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_pca_mean_var[0], label="ICA", color = "c")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "c")
    else: 
        plt.plot(dims_list, transf_pca_mean_var[0], label="ICA", color = "c")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.6, color = "c", linewidth = 2)
        
    # Plot for tICA for reconstructed data
    print(transf_tica_mean_var)
    upper_var = np.array(transf_tica_mean_var[0]) + np.array(transf_tica_mean_var[1])
    lower_var = np.array(transf_tica_mean_var[0]) - np.array(transf_tica_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_tica_mean_var[0], label="tICA", color = "b")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "b")
    else: 
        plt.plot(dims_list, transf_tica_mean_var[0], label="tICA", color = "b")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.6, color = "b", linewidth = 2)

    # Plot for Vector ICA for transformed data
    upper_var = np.array(transf_vector_mean_var[0]) + np.array(transf_vector_mean_var[1])
    lower_var = np.array(transf_vector_mean_var[0]) - np.array(transf_vector_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_vector_mean_var[0], label="Vector-ICA", color = "orange")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "orange")
    else: 
        plt.plot(dims_list, transf_vector_mean_var[0], label="Vector-ICA", color = "orange")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.6, color = "orange", linewidth = 2)

    plt.xlabel("Number of dimensions reduced to " + str(target_dim))
    plt.ylabel("Average deviation from original data")
    plt.title("Transformed: Average devation vs. dynamic dimension for original space")
    plt.legend()
    plt.show()


def plot_ica_reconstr_dyn_low_dims_vs_avg_dev(seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False):
    '''
    Plot the average deviation vs. the dynamic changing lower dimension in the original/ reconstructed space. x-axis: number of dimensions to which start_dim is transformed and then reconstructed.

    Parameters:
    start_dim (integer): specifies what dimension the data is to begin with
    order (string): data is generated for every higher dimension in specified order
    bounds (tuple(integer)): data is generated for every higher dimension in specified bounds
    num_lines (integer): number of lines which are generated for specified order
    num_points_per_line (integer): number of points on each line
    dist_measure (string): euclidean or mahalanobis distance measure
    k_neigh (integer): number how many neighbors should be compared
    sep_measure (string): specifies if lines are supposed to measured separately. Only for orders suitable for this!
    jitter (integer): how far the points are laying from the lines

    Returns: 
    nothing (plot)
    '''

    pca_mean_var, vector_mean_var, _, _, _ = compute_ica_dyn_low_dims_vs_avg_dev(seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end)

    dim_list = generate_integer_list_dyn_low(start_dim, num_lines, order)

    # Plot for ICA for reconstructed data
    upper_var = np.array(pca_mean_var[0]) + np.array(pca_mean_var[1])
    lower_var = np.array(pca_mean_var[0]) - np.array(pca_mean_var[1])

    if sep_measure:
        plt.plot(dim_list, pca_mean_var[0], label="Expected Value: ICA", color = "c")
        plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.3, color = "c")
    else: 
        plt.plot(dim_list, pca_mean_var[0], label="Expected Value: PCA", color = "c")
        plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.6, color = "c", linewidth = 2)

    # Plot for Vector ICA for reconstructed data
    upper_var = np.array(vector_mean_var[0]) + np.array(vector_mean_var[1])
    lower_var = np.array(vector_mean_var[0]) - np.array(vector_mean_var[1])
    
    if sep_measure:
        plt.plot(dim_list, vector_mean_var[0], label="Expected Value: Vector PCA", color = "orange")
        plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.3, color = "orange")
    else:
        plt.plot(dim_list, vector_mean_var[0], label="Expected Value: Vector PCA", color = "orange")
        plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.6, color = "orange", linewidth = 2)

    plt.xlabel("number of dimensions to which dimension " + str(start_dim) + " is reduced and reconstructed")
    plt.ylabel("Average deviation from original data")
    plt.title("Reconstructed: Average devation vs. dynamic dimension for projected space")
    plt.legend()
    plt.show()   


def plot_ica_transf_dyn_low_dims_vs_avg_dev(seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False):
    
    _, _, transf_pca_mean_var, transf_vector_mean_var, transf_tica_mean_var = compute_ica_dyn_low_dims_vs_avg_dev(seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end)

    dims_list = generate_integer_list_dyn_low(start_dim, num_lines, order)

    # Plot for ICA for reconstructed data
    upper_var = np.array(transf_pca_mean_var[0]) + np.array(transf_pca_mean_var[1])
    lower_var = np.array(transf_pca_mean_var[0]) - np.array(transf_pca_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_pca_mean_var[0], label="ICA", color = "c")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "c")
    else: 
        plt.plot(dims_list, transf_pca_mean_var[0], label="ICA", color = "c")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.6, color = "c", linewidth = 2)
    
     # Plot for tICA for reconstructed data
    upper_var = np.array(transf_tica_mean_var[0]) + np.array(transf_tica_mean_var[1])
    lower_var = np.array(transf_tica_mean_var[0]) - np.array(transf_tica_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_tica_mean_var[0], label="tICA", color = "b")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "b")
    else: 
        plt.plot(dims_list, transf_tica_mean_var[0], label="tICA", color = "b")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.6, color = "b", linewidth = 2)

    # Plot for Vector ICA for transformed data
    upper_var = np.array(transf_vector_mean_var[0]) + np.array(transf_vector_mean_var[1])
    lower_var = np.array(transf_vector_mean_var[0]) - np.array(transf_vector_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_vector_mean_var[0], label="Expected Value: Original", color = "orange")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "orange")
    else: 
        plt.plot(dims_list, transf_vector_mean_var[0], label="Expected Value: Original", color = "orange")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.6, color = "orange", linewidth = 2)

    plt.xlabel("number of dimensions to which dimension " + str(start_dim) + " is reduced")
    plt.ylabel("Average deviation from original data")
    plt.title("Transformed: Average devation vs. dynamic dimension for projected space")
    plt.legend()
    plt.show()


def plot_ica_vs_vector_ica_rec_neighborh_error(picked_options, k, sep_measure):
    """
    Plot average deviation vs. increasing number of neighbours.

    Parameters:
        picked_options (dictionary): Input data matrix with shape (n_samples, n_features).
        k (int): Maximum number of dimensions to consider for dimensionality reduction.

    Returns:
        List of means and variances for pca, vector_pca and random_projection
    """
     
    separated = False

    # Get the correct data structure
    sep_points = picked_options["sep_points"]
    separated = picked_options["separated"]

    complex_data = add_complex_vector(sep_points, separated)
    id_data = add_id(sep_points, separated)
    vector_data = np.imag(complex_data)

    if separated:
        data1 = flatten(picked_options["sep_points"])
        id_data = flatten(id_data)
    else:
        data1 = sep_points
        id_data = id_data
    
    # Compute the transformed and reconstructed data for original, complex and ID data
    ica = FastICA(n_components=picked_options["target_dim"])
    ica_transf = ica.fit_transform(data1)
    ica_reconstr = ica.inverse_transform(ica_transf)

    vector_ica = FastICA(n_components=picked_options["target_dim"])
    vector_ica.fit(vector_data)
    vector_ica_transf = vector_ica.transform(data1)
    vector_ica_reconstr = vector_ica.inverse_transform(vector_ica_transf)

    # Add structure name to legend
    plt.plot([], [], ' ', label="structure " + str(picked_options["order"]))

    # Compute mean and std for original data
    # Compute mean and std for complex transformed data
    # Compute mean and std for ID data
    if not sep_measure:
        exp_values, vars, median = get_mu_var_by_neighbor_num(data1, ica_reconstr, picked_options["dist_measure"], k)
        exp_comp_values, comp_vars, comp_median = get_mu_var_by_neighbor_num(data1, vector_ica_reconstr, picked_options["dist_measure"], k)
    elif sep_measure:
        sep_rec_data = [ica_reconstr[i:i + picked_options["num_points"]] for i in range(0, len(ica_reconstr), picked_options["num_points"])]
        sep_rec_comp_data = [vector_ica_reconstr[i:i + picked_options["num_points"]] for i in range(0, len(vector_ica_reconstr), picked_options["num_points"])]

        exp_values, vars, median = measure_sep_structures(picked_options["sep_points"], sep_rec_data, picked_options["dist_measure"], k)
        exp_comp_values, comp_vars, comp_median = measure_sep_structures(picked_options["sep_points"], sep_rec_comp_data, picked_options["dist_measure"], k)

    # Plot for original PCA
    upper_var = np.array(exp_values) + np.array(vars)
    lower_var = np.array(exp_values) - np.array(vars)

    if sep_measure:
        plt.plot(range(min(k, picked_options["num_points"])), median, label="Median: ICA", color = "r", linestyle="dotted")
        plt.plot(range(min(k, picked_options["num_points"])), exp_values, label="Expected Value: ICA", color = "c")
        plt.fill_between(range(min (picked_options["num_points"], k)), lower_var, upper_var, alpha = 0.3, color = "c")
    else:
        plt.plot(range(min(len(data1), k)), median, label="Median: ICA", color = "c", linestyle="dotted")
        plt.plot(range(min(len(data1), k)), exp_values, label="Expected Value: ICA", color = "c")
        plt.fill_between(range(min(len(data1), k)), lower_var, upper_var, alpha = 0.6, color = "c", linewidth = 2)

    # Plot for complex PCA
    comp_lower_var = np.array(exp_comp_values) - np.array(comp_vars)
    comp_upper_var = np.array(exp_comp_values) + np.array(comp_vars)

    if sep_measure:
        plt.plot(range(min(k, picked_options["num_points"])), comp_median, label="Median: Vector ICA", color = "orange", linestyle="dotted")
        plt.plot(range(min(k, picked_options["num_points"])), exp_comp_values, label="Expected Value: Vector ICA", color = "orange")
        plt.fill_between(range(min(picked_options["num_points"], k)), comp_lower_var, comp_upper_var, alpha = 0.3, color = "orange")
    else:
        plt.plot(range(min(k, len(data1))), comp_median, label="Median: Vector ICA", color = "orange", linestyle="dotted")
        plt.plot(range(min(k, len(data1))), exp_comp_values, label="Expected Value: Vector ICA", color = "orange")
        plt.fill_between(range(min(len(data1), k)), comp_lower_var, comp_upper_var, alpha = 0.3, color = "orange")
    
    plt.legend()
    plt.title("ICA Structure Preservation: Mean & Variance in higher dim space")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Average deviation from original")
    plt.show()


def plot_ica_vs_vector_ica_transf_neighborh_error(picked_options, k, sep_measure):
    """
    Plot average deviation vs. increasing number of neighbours.

    Parameters:
        picked_options (dictionary): Input data matrix with shape (n_samples, n_features).
        k (int): Maximum number of dimensions to consider for dimensionality reduction.

    Returns:
        List of means and variances for pca, vector_pca and random_projection
    """
     
    separated = False

    # Get the correct data structure
    sep_points = picked_options["sep_points"]
    separated = picked_options["separated"]

    complex_data = add_complex_vector(sep_points, separated)
    id_data = add_id(sep_points, separated)
    vector_data = get_neighbor_vector(sep_points, separated)

    if separated:
        data1 = flatten(picked_options["sep_points"])
        id_data = flatten(id_data)
    else:
        data1 = sep_points
        id_data = id_data
    
    # Compute the transformed and reconstructed data for original, complex and ID data
    ica = FastICA(n_components=picked_options["target_dim"])
    ica_transf = ica.fit_transform(data1)

    vector_ica = FastICA(n_components=picked_options["target_dim"])
    vector_ica.fit(vector_data)
    vector_ica_transf = vector_ica.transform(data1)

    # Add structure name to legend
    plt.plot([], [], ' ', label="structure " + str(picked_options["order"]))

    # Compute mean and std for original data
    # Compute mean and std for complex transformed data
    # Compute mean and std for ID data
    if not sep_measure:
        exp_values, vars, median = get_mu_var_by_neighbor_num(data1, ica_transf, picked_options["dist_measure"], k)
        exp_comp_values, comp_vars, comp_median = get_mu_var_by_neighbor_num(data1, vector_ica_transf, picked_options["dist_measure"], k)
    elif sep_measure:
        sep_rec_data = [ica_transf[i:i + picked_options["num_points"]] for i in range(0, len(ica_transf), picked_options["num_points"])]
        sep_rec_comp_data = [vector_ica_transf[i:i + picked_options["num_points"]] for i in range(0, len(vector_ica_transf), picked_options["num_points"])]

        exp_values, vars, median = measure_sep_structures(picked_options["sep_points"], sep_rec_data, picked_options["dist_measure"], k)
        exp_comp_values, comp_vars, comp_median = measure_sep_structures(picked_options["sep_points"], sep_rec_comp_data, picked_options["dist_measure"], k)

    # Plot for original PCA
    upper_var = np.array(exp_values) + np.array(vars)
    lower_var = np.array(exp_values) - np.array(vars)

    if sep_measure:
        plt.plot(range(min(k, picked_options["num_points"])), median, label="Median: ICA", color = "r", linestyle="dotted")
        plt.plot(range(min(k, picked_options["num_points"])), exp_values, label="Expected Value: ICA", color = "c")
        plt.fill_between(range(min (picked_options["num_points"], k)), lower_var, upper_var, alpha = 0.3, color = "c")
    else:
        plt.plot(range(min(len(data1), k)), median, label="Median: ICA", color = "c", linestyle="dotted")
        plt.plot(range(min(len(data1), k)), exp_values, label="Expected Value: ICA", color = "c")
        plt.fill_between(range(min(len(data1), k)), lower_var, upper_var, alpha = 0.6, color = "c", linewidth = 2)

    # Plot for complex PCA
    comp_lower_var = np.array(exp_comp_values) - np.array(comp_vars)
    comp_upper_var = np.array(exp_comp_values) + np.array(comp_vars)

    if sep_measure:
        plt.plot(range(min(k, picked_options["num_points"])), comp_median, label="Median: Vector ICA", color = "orange", linestyle="dotted")
        plt.plot(range(min(k, picked_options["num_points"])), exp_comp_values, label="Expected Value: Vector ICA", color = "orange")
        plt.fill_between(range(min(picked_options["num_points"], k)), comp_lower_var, comp_upper_var, alpha = 0.3, color = "orange")
    else:
        plt.plot(range(min(k, len(data1))), comp_median, label="Median: Vector ICA", color = "orange", linestyle="dotted")
        plt.plot(range(min(k, len(data1))), exp_comp_values, label="Expected Value: Vector ICA", color = "orange")
        plt.fill_between(range(min(len(data1), k)), comp_lower_var, comp_upper_var, alpha = 0.3, color = "orange")
    
    plt.legend()
    plt.title("Structure Preservation: Mean & Variance in transformed dim space")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Average deviation from original")
    plt.show()



