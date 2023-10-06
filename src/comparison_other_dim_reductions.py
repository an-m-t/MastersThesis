from sklearn.manifold import LocallyLinearEmbedding, TSNE
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from structure_measures import *
import matplotlib.pyplot as plt
from data_generation import *
from general_func import pickle_results_avg_dev_vs_dyn_dim
from general_func import *
from complex_PCA import *
import numpy as np
import pickle


def pickle_lle_tsne_umap_results_avg_dev_vs_dyn_dim(file_path, parameters, transf_lle_mean_var, transf_tsne_mean_var, transf_umap_mean_var):
    new_result = {
        "parameters" : parameters,
        "transf_lle_mean_var" : transf_lle_mean_var,
        "transf_tsne_mean_var" : transf_tsne_mean_var,
        "transf_umap_mean_var" : transf_umap_mean_var
    }
    try:
        with open(file_path, 'rb') as file:
            prev_results = pickle.load(file)
    except:
        prev_results = {}
    
    prev_results[len(prev_results) + 1] = new_result
    
    with open(file_path, 'wb') as file:
        pickle.dump(prev_results, file)
        

def find_lle_tsne_umap_in_pickle(file_path, parameters):
    try:
        with open(file_path, 'rb') as file:
            results = pickle.load(file)
    except:
        print("File does not exist or is empty")
        return None
    for i, result in results.items():
        if result["parameters"] == parameters:
            print("Found Run")
            return result["transf_lle_mean_var"], result["transf_tsne_mean_var"], result["transf_umap_mean_var"]
        
    print("Result not yet pickled")
    return None

def find_lle_tsne_umap_in_pickle_wo_target_dim(file_path, parameters_wo_target_dim):
    try:
        with open(file_path, 'rb') as file:
            results = pickle.load(file)
    except:
        print("File does not exist or is empty")
        return None
    result_list = []
    print("lle umap tsne ", parameters_wo_target_dim)
    for i, result in results.items():
        print(result["parameters"])
        if result["parameters"][:-1] == parameters_wo_target_dim:
            print("Found Run")
            result_list.append(result)
        
    return result_list


def compute_lle_tsne_umap_dyn_high_dims_vs_avg_dev(seed, target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False):
    """
    Compute average deviation vs. dimensionality of the data for dynamically increasing number of dimensions for the original space.

    Parameters:
        data (array-like): Input data matrix with shape (n_samples, n_features).
        target_dim (int): Number of dimensions specified for transformation

    Returns:
        List of means and variances for pca, vector_pca and random_projection
    """
    
    pickled_lle_tsne = find_lle_tsne_umap_in_pickle("./pickled_results/other_methods_avg_dev_vs_dyn_high_dims.pickle", [seed, target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end])
    
    if pickled_lle_tsne is not None:
        return pickled_lle_tsne
    else:
        dim_list = general_func.generate_integer_list_dyn_high(target_dim)
        separated = False

        list_lle_transf_means = []
        list_lle_transf_vars = []

        list_tsne_transf_means = []
        list_tsne_transf_vars = []

        list_umap_transf_means = []
        list_umap_transf_vars = []

        for start_dimension in dim_list:
            sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dimension, jitter, bounds, parallel_start_end)
            
            if separated:
                data = flatten(sep_points)
            else:
                data = sep_points

            # Transform data with lle
            try:
                lle = LocallyLinearEmbedding(n_components=target_dim)
                lle_transf_data = lle.fit_transform(data)
            except:
                lle = LocallyLinearEmbedding(n_components=target_dim, eigen_solver="dense")
                lle_transf_data = lle.fit_transform(data)

            # Transform data with t-sne
            try:
                tsne = TSNE(n_components=target_dim)
                tsne_transf_data = tsne.fit_transform(data)
            except:
                tsne = TSNE(n_components=target_dim, method="exact")
                tsne_transf_data = tsne.fit_transform(data)

            # Transform data with umap
            umap_reducer = UMAP(n_components=target_dim)
            scaled_data = StandardScaler().fit_transform(data)
            umap_transf_data = umap_reducer.fit_transform(scaled_data)
        
            # Compute exp-values and variances for lle and t-sne transformed data
            if not sep_measure:
                print("#####")
                lle_transf_exp_values, lle_transf_vars, _ = get_mu_var_by_neighbor_num(data, lle_transf_data, dist_measure, k_neigh)
                print(lle_transf_exp_values)
                tsne_transf_exp_values, tsne_transf_vars, _ = get_mu_var_by_neighbor_num(data, tsne_transf_data, dist_measure, k_neigh)
                umap_transf_exp_values, umap_transf_vars, _ = get_mu_var_by_neighbor_num(data, umap_transf_data, dist_measure, k_neigh)

            elif sep_measure:
                sep_lle_transf_data = [lle_transf_data[i:i + num_points_per_line] for i in range(0, len(lle_transf_data), num_points_per_line)]
                sep_tsne_transf_data = [tsne_transf_data[i:i + num_points_per_line] for i in range(0, len(tsne_transf_data), num_points_per_line)]
                sep_umap_transf_data = [umap_transf_data[i:i + num_points_per_line] for i in range(0, len(umap_transf_data), num_points_per_line)]

                lle_transf_exp_values, lle_transf_vars, _ = measure_sep_structures(sep_points, sep_lle_transf_data, dist_measure, k_neigh)
                tsne_transf_exp_values, tsne_transf_vars, _ = measure_sep_structures(sep_points, sep_tsne_transf_data, dist_measure, k_neigh)
                umap_transf_exp_values, umap_transf_vars, _ = measure_sep_structures(sep_points, sep_umap_transf_data, dist_measure, k_neigh)

            # Append
            list_lle_transf_means.append(lle_transf_exp_values[-1])
            list_lle_transf_vars.append(lle_transf_vars[-1])

            list_tsne_transf_means.append(tsne_transf_exp_values[-1])
            list_tsne_transf_vars.append(tsne_transf_vars[-1])

            list_umap_transf_means.append(umap_transf_exp_values[-1])
            list_umap_transf_vars.append(umap_transf_vars[-1])
        
        pickle_lle_tsne_umap_results_avg_dev_vs_dyn_dim("./pickled_results/other_methods_avg_dev_vs_dyn_high_dims.pickle", [seed, target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end, dim_list], (list_lle_transf_means, list_lle_transf_vars), (list_tsne_transf_means, list_tsne_transf_vars), (list_umap_transf_means, list_umap_transf_vars))

        return[(list_lle_transf_means, list_lle_transf_vars), (list_tsne_transf_means, list_tsne_transf_vars), (list_umap_transf_means, list_umap_transf_vars)]


def compute_lle_tsne_dyn_low_dims_vs_avg_dev(seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False, data = None):
    """
    Compute average deviation vs. dimensionality of the data for dynamically increasing number of dimensions for transformed space.

    Parameters:
        data (array-like): Input data matrix with shape (n_samples, n_features).
        start_dim (int): Number of dimensions specified for original data

    Returns:
        List of means and variances for pca, vector_pca and random_projection
    """
    
    pickled_lle_transf = find_lle_tsne_umap_in_pickle("./pickled_results/other_methods_avg_dev_vs_dyn_low_dims.pickle", [seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end])

    if pickled_lle_transf is not None:
        return pickled_lle_transf
    
    else:

        dim_list = general_func.generate_integer_list_dyn_low(start_dim, num_lines, order)
        separated = False

        list_transf_lle_means = []
        list_transf_lle_vars = []

        list_transf_tsne_means = []
        list_transf_tsne_vars= []

        list_umap_transf_means = []
        list_umap_transf_vars = []

        sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dim, jitter, bounds, parallel_start_end)
            
        if separated:
            data = flatten(sep_points)
        else:
            data = sep_points

        for target_dim in dim_list:
            # Transform with lle
            try:
                lle = LocallyLinearEmbedding(n_components=target_dim)
                lle_transf_data = lle.fit_transform(data)
                break
            except:
                lle = LocallyLinearEmbedding(n_components=target_dim, eigensolver="dense")
                lle_transf_data = lle.fit_transform(data)
            

            # Transform with tsne
            try:
                tsne = TSNE(n_components=target_dim)
                tsne_transf_data = tsne.fit_transform(data)
            except:
                tsne = TSNE(n_components=target_dim, method="exact")
                tsne_transf_data = tsne.fit_transform(data)

            # Transform data with umap
            umap_reducer = UMAP(n_components=target_dim)
            scaled_data = StandardScaler().fit_transform(data)
            umap_transf_data = umap_reducer.fit_transform(scaled_data)

            # Transform data
            # Compute exp-values and variances for reconstructed and transformed data
            if not sep_measure:
                transf_lle_exp_values, transf_lle_vars, _ = get_mu_var_by_neighbor_num(data, lle_transf_data, dist_measure, k_neigh)
                transf_tsne_exp_values, transf_tsne_vars, _ = get_mu_var_by_neighbor_num(data, tsne_transf_data.real, dist_measure, k_neigh)
                umap_transf_exp_values, umap_transf_vars, _ = get_mu_var_by_neighbor_num(data, umap_transf_data, dist_measure, k_neigh)

            elif sep_measure:
                sep_transf_lle_data = [lle_transf_data[i:i + num_points_per_line] for i in range(0, len(lle_transf_data), num_points_per_line)]
                sep_transf_tsne_data = [tsne_transf_data[i:i + num_points_per_line] for i in range(0, len(tsne_transf_data), num_points_per_line)]
                sep_umap_transf_data = [umap_transf_data[i:i + num_points_per_line] for i in range(0, len(umap_transf_data), num_points_per_line)]

                transf_lle_exp_values, transf_lle_vars, _ = measure_sep_structures(sep_points, sep_transf_lle_data, dist_measure, k_neigh)
                transf_tsne_exp_values, transf_tsne_vars, _ = measure_sep_structures(sep_points, sep_transf_tsne_data, dist_measure, k_neigh)
                umap_transf_exp_values, umap_transf_vars, _ = measure_sep_structures(sep_points, sep_umap_transf_data, dist_measure, k_neigh)

            # Append
            list_transf_lle_means.append(transf_lle_exp_values[-1])
            list_transf_tsne_means.append(transf_tsne_exp_values[-1])

            list_transf_lle_vars.append(transf_lle_vars[-1])
            list_transf_tsne_vars.append(transf_tsne_vars[-1])

            list_umap_transf_means.append(umap_transf_exp_values[-1])
            list_umap_transf_vars.append(umap_transf_vars[-1])
        
        pickle_lle_tsne_umap_results_avg_dev_vs_dyn_dim("./pickled_results/other_methods_avg_dev_vs_dyn_low_dims.pickle", [seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end, dim_list], (list_transf_lle_means, list_transf_lle_vars), (list_transf_tsne_means, list_transf_tsne_vars), (list_umap_transf_means, list_umap_transf_vars))
        
        return (list_transf_lle_means, list_transf_lle_vars), (list_transf_tsne_means, list_transf_tsne_vars), (list_umap_transf_means, list_umap_transf_vars)


def plot_lle_tsne_transf_dyn_high_dims_vs_avg_dev(seed, target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False):
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

    _, _, _, transf_pca_mean_var, transf_vector_mean_var, transf_random_proj_mean_var = compute_dyn_high_dims_vs_avg_dev(seed, target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end)

    transf_lle_mean_var, transf_tsne_mean_var, transf_umap_mean_var = compute_lle_tsne_umap_dyn_high_dims_vs_avg_dev(seed, target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end)

    dims_list = generate_integer_list_dyn_high(target_dim)

    # Plot for PCA for transformed data
    upper_var = np.array(transf_pca_mean_var[0]) + np.array(transf_pca_mean_var[1])
    lower_var = np.array(transf_pca_mean_var[0]) - np.array(transf_pca_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_pca_mean_var[0], label="PCA", color = "c")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "c")
    else: 
        plt.plot(dims_list, transf_pca_mean_var[0], label="PCA", color = "c")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.6, color = "c", linewidth = 2)

    # Plot for Vector PCA for transformed data
    upper_var = np.array(transf_vector_mean_var[0]) + np.array(transf_vector_mean_var[1])
    lower_var = np.array(transf_vector_mean_var[0]) - np.array(transf_vector_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_vector_mean_var[0], label="PCA*", color = "orange")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "orange")
    else: 
        plt.plot(dims_list, transf_vector_mean_var[0], label="PCA*", color = "orange")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.6, color = "orange", linewidth = 2)

    # Plot for lle for transformed data
    upper_var = np.array(transf_lle_mean_var[0]) + np.array(transf_lle_mean_var[1])
    lower_var = np.array(transf_lle_mean_var[0]) - np.array(transf_lle_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_lle_mean_var[0], label="LLE", color = "m")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "m")
    else: 
        plt.plot(dims_list, transf_lle_mean_var[0], label="LLE", color = "m")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "m", linewidth = 2)

    # Plot for tsne for transformed data
    upper_var = np.array(transf_tsne_mean_var[0]) + np.array(transf_tsne_mean_var[1])
    lower_var = np.array(transf_tsne_mean_var[0]) - np.array(transf_tsne_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_tsne_mean_var[0], label="T-SNE", color = "pink")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "pink")
    else: 
        plt.plot(dims_list, transf_tsne_mean_var[0], label="T-SNE", color = "pink")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "pink", linewidth = 2)

    # Plot for tsne for transformed data
    upper_var = np.array(transf_umap_mean_var[0]) + np.array(transf_umap_mean_var[1])
    lower_var = np.array(transf_umap_mean_var[0]) - np.array(transf_umap_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_umap_mean_var[0], label="Expected Value: UMAP", color = "brown")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "brown")
    else: 
        plt.plot(dims_list, transf_umap_mean_var[0], label="Expected Value: UMAP", color = "brown")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "brown", linewidth = 2)

    plt.xlabel("Number of dimensions reduced to " + str(target_dim))
    plt.ylabel("Average deviation from original data")
    # plt.title("Transformed: Average deviation vs. dynamic dimension for original space")
    # plt.suptitle(str(num_lines) + " number of lines a " + str(num_points_per_line) + " points per line and comparing " + str(k_neigh) + " neighbours")
    plt.savefig("plots/generated/" + str(order) + "/avg_dev_vs_dyn_low/" + str(num_lines) + "lines_" + str(num_points_per_line) + "points_" + str(dist_measure) + "measure.pdf")
    plt.legend()

    plt.show()


def plot_lle_tsne_transf_dyn_low_dims_vs_avg_dev(seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False, data = None):

    _, _, _, transf_pca_mean_var,  transf_vector_mean_var, _ = compute_dyn_low_dims_vs_avg_dev(seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end)

    transf_lle_mean_var, transf_tsne_mean_var, transf_umap_mean_var = compute_lle_tsne_dyn_low_dims_vs_avg_dev(seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end)
    
    dims_list = generate_integer_list_dyn_low(start_dim, num_lines, order)

    # Plot for PCA for reconstructed data
    upper_var = np.array(transf_pca_mean_var[0]) + np.array(transf_pca_mean_var[1])
    lower_var = np.array(transf_pca_mean_var[0]) - np.array(transf_pca_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_pca_mean_var[0], label="PCA", color = "c")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "c")
    else: 
        plt.plot(dims_list, transf_pca_mean_var[0], label="PCA", color = "c")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.6, color = "c", linewidth = 2)

    # Plot for Vector PCA for transformed data
    upper_var = np.array(transf_vector_mean_var[0]) + np.array(transf_vector_mean_var[1])
    lower_var = np.array(transf_vector_mean_var[0]) - np.array(transf_vector_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_vector_mean_var[0], label="Vector PCA", color = "orange")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "orange")
    else: 
        plt.plot(dims_list, transf_vector_mean_var[0], label="Vector PCA", color = "orange")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.6, color = "orange", linewidth = 2)

    # Plot for lle projection for transformed data
    upper_var = np.array(transf_lle_mean_var[0]) + np.array(transf_lle_mean_var[1])
    lower_var = np.array(transf_lle_mean_var[0]) - np.array(transf_lle_mean_var[1])
    if sep_measure:
        plt.plot(dims_list, transf_lle_mean_var[0], label="Expected Value: LLE", color = "m")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "m")
    else: 
        plt.plot(dims_list, transf_lle_mean_var[0], label="Expected Value: LLE", color = "m")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "m", linewidth = 2)

    # Plot for tsne projection for transformed data
    upper_var = np.array(transf_tsne_mean_var[0]) + np.array(transf_tsne_mean_var[1])
    lower_var = np.array(transf_tsne_mean_var[0]) - np.array(transf_tsne_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_tsne_mean_var[0], label="Expected Value: T-SNE", color = "pink")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "pink")
    else: 
        plt.plot(dims_list, transf_tsne_mean_var[0], label="Expected Value: T-SNE", color = "pink")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "pink", linewidth = 2)

    # Plot for tsne projection for transformed data
    upper_var = np.array(transf_umap_mean_var[0]) + np.array(transf_umap_mean_var[1])
    lower_var = np.array(transf_umap_mean_var[0]) - np.array(transf_umap_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_umap_mean_var[0], label="Expected Value: UMAP", color = "brown")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "brown")
    else: 
        plt.plot(dims_list, transf_umap_mean_var[0], label="Expected Value: UMAP", color = "brown")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "brown", linewidth = 2)

    plt.xlabel("number of dimensions to which dimension " + str(start_dim) + " is reduced")
    plt.ylabel("Average deviation from original data")
    plt.title("Transformed: Average devation vs. dynamic dimension for projected space")
    plt.suptitle(str(num_lines) + " number of lines a " + str(num_points_per_line) + " points per line and comparing " + str(k_neigh) + " neighbours")
    plt.legend()
    
    plt.savefig("plots/generated/"+ str(order) + "/avg_dev_vs_dyn_low/" + str(num_lines) + "lines_" + str(num_points_per_line) + "points_" + str(dist_measure) + "measure.pdf")
    plt.show()