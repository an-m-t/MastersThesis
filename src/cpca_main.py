from comparison_other_dim_reductions import *
from plot_func.dynamic_dimensions import *
from plot_func.plotting import *
from plot_func.heatmaps import *
from data_generation import *
from data_generation import add_id
# import partial_order
import complex_PCA
import random
from ica import *
from real_data_cleaning import *
from multiple_runs import *
import logging
import sys
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
import matplotlib.ticker as tkr
from matplotlib import rcParams
rcParams['axes.titlepad'] = 20 
matplotlib.rcParams.update({'font.size': 18})


plot_dict = {
    "lineplots" : {
        "fontsize" : 36,
        "font_dist": 6,
        "font_legend": 23,
        "dtw_fontsize": 33,
        "dtw_fontdist": 5,
        "dtw_font_legend": 30
    },
    "heatmaps" : {
        "fontsize" : 21,
        "font_dist" : 0.5
    }
}

def cpca_multiple_avg_dev_vs_dyn_low_dim(start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, num_runs, parallel_start_end = False):
    """
    Compute average deviation vs. dimensionality of the data for dynamically increasing number of dimensions for transformed space.

    Parameters:
        data (array-like): Input data matrix with shape (n_samples, n_features).
        start_dim (int): Number of dimensions specified for original data

    Returns:
        List of means and variances for pca, vector_pca and random_projection
    """
    logging.info("Start computing dynamic dims vs avg dev")
    
    seed_list = range(num_runs)
    
    dim_list = general_func.generate_integer_list_dyn_low(start_dim, num_lines, order)
    
    for s in seed_list:
        slurm_pickled = find_in_pickle_dyn_dims("./pickled_results/slurm/cpca_avg_dev_vs_dyn_low_dims.pickle", [s, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end, dim_list])
        
        pickled = find_in_pickle_dyn_dims("./pickled_results/cpca_avg_dev_vs_dyn_low_dims.pickle", [s, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end, dim_list])

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

            list_cpca_real_pca_means = []
            list_cpca_real_pca_vars = []

            list_cpca_real_low_means = []
            list_cpca_real_low_vars = []

            list_cpca_imag_means = []
            list_cpca_imag_vars = []
            
            list_id_means = []
            list_id_vars = []
            
            list_transf_pca_means = []
            list_transf_pca_vars = []
            
            list_transf_cpca_real_pca_means = []
            list_transf_cpca_real_pca_vars = []

            list_transf_cpca_real_low_means = []
            list_transf_cpca_real_low_vars = []

            list_transf_cpca_imag_means = []
            list_transf_cpca_imag_vars = []
            
            list_transf_id_means = []
            list_transf_id_vars = []

            # Generate data in current dimensionality
            sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dim, jitter, bounds, parallel_start_end = False, standardise = False)
            if separated:
                data = flatten(sep_points)
            else:
                data = sep_points

            for target_dim in dim_list:
                logging.info("Transform and reconstruct data with pca for target_dim" + str(target_dim))
                reconstr_data, transf_data = pca_reconstruction(data, target_dim, "complex_pca", None, None)
                
                complex_data = add_complex_vector(sep_points, separated)

                # Transform and reconstruct data with cpca and scale to original pca
                logging.info("Get vector data")
                reconstr_cpca_real_pca_data, transf_cpca_real_pca_data = complex_PCA.pca_reconstruction(complex_data, target_dim, "complex_pca_svd_complex", "like_original_pca", None)
                
                # Transform and reconstruct data with cpca and scale to lowest imaginary part
                logging.info("Get vector data")
                reconstr_cpca_real_low_data, transf_cpca_real_low_data = complex_PCA.pca_reconstruction(complex_data, target_dim, "complex_pca_svd_complex", "low_imag", None)
                
                # Transform and reconstruct data with cpca and scale to lowest imaginary part
                logging.info("Get vector data")
                reconstr_cpca_imag_data, transf_cpca_imag_data = complex_PCA.pca_reconstruction(complex_data, target_dim, "complex_pca_svd_complex_imag_pc", "low_imag", None)
                
                # Add Ids and reconstruct and transform data
                if separated:
                    id_data = flatten(add_id(sep_points, separated))
                else:
                    id_data = add_id(sep_points, separated)
                reconstr_id_data, transf_id_data = pca_reconstruction(id_data, target_dim + 1, "complex_pca", None, None)
                reconstr_id_data = [elem[:-1] for elem in reconstr_id_data]
                transf_id_data = [elem[:-1] for elem in transf_id_data]

                # Compute exp-values and variances for reconstructed and transformed data
                logging.info("Compute expected values, variances and median")
                if not sep_measure:
                    reconstr_exp_values, reconstr_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_data, dist_measure, k_neigh)
                    
                    reconstr_exp_cpca_real_pca_values, reconstr_cpca_real_pca_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_cpca_real_pca_data, dist_measure, k_neigh)
                    reconstr_exp_cpca_low_values, reconstr_cpca_low_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_cpca_real_low_data, dist_measure, k_neigh)
                    reconstr_exp_cpca_imag_values, reconstr_cpca_imag_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_cpca_imag_data, dist_measure, k_neigh)
                    reconstr_exp_id_values, reconstr_id_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_id_data, dist_measure, k_neigh)

                    if dist_measure != "dtw":
                        transf_exp_values, transf_vars, _ = get_mu_var_by_neighbor_num(data, transf_data, dist_measure, k_neigh)
                        transf_exp_cpca_real_pca_values, transf_cpca_real_pca_vars, _ = get_mu_var_by_neighbor_num(data, transf_cpca_real_pca_data, dist_measure, k_neigh)
                        transf_exp_cpca_low_values, transf_cpca_low_vars, _ = get_mu_var_by_neighbor_num(data, transf_cpca_real_low_data, dist_measure, k_neigh)
                        transf_exp_cpca_imag_values, transf_cpca_imag_vars, _ = get_mu_var_by_neighbor_num(data, transf_cpca_imag_data, dist_measure, k_neigh)
                        transf_exp_id_values, transf_id_vars, _ = get_mu_var_by_neighbor_num(data, transf_id_data, dist_measure, k_neigh)

                elif sep_measure:
                    sep_rec_data = [reconstr_data[i:i + num_points_per_line] for i in range(0, len(reconstr_data), num_points_per_line)]
                    sep_rec_cpca_real_pca_data = [reconstr_cpca_real_pca_data[i:i + num_points_per_line] for i in range(0, len(reconstr_cpca_real_pca_data), num_points_per_line)]
                    sep_rec_cpca_real_low_data = [reconstr_cpca_real_low_data[i:i + num_points_per_line] for i in range(0, len(reconstr_cpca_real_low_data), num_points_per_line)]
                    sep_rec_cpca_imag_data = [reconstr_cpca_imag_data[i:i + num_points_per_line] for i in range(0, len(reconstr_cpca_imag_data), num_points_per_line)]
                    sep_rec_cpca_id_data = [reconstr_id_data[i:i + num_points_per_line] for i in range(0, len(reconstr_id_data), num_points_per_line)]

                    if dist_measure != "dtw":
                        reconstr_exp_values, reconstr_vars, _ = measure_sep_structures(sep_points, sep_rec_data, dist_measure, k_neigh)
                        reconstr_exp_cpca_real_pca_values, reconstr_cpca_real_pca_vars, _ = measure_sep_structures(sep_points, sep_rec_cpca_real_pca_data, dist_measure, k_neigh)
                        reconstr_exp_cpca_low_values, reconstr_cpca_low_vars, _ = measure_sep_structures(sep_points, sep_rec_cpca_real_low_data, dist_measure, k_neigh)
                        reconstr_exp_cpca_imag_values, reconstr_cpca_imag_vars, _ = measure_sep_structures(sep_points, sep_rec_cpca_imag_data, dist_measure, k_neigh)
                        reconstr_exp_id_values, reconstr_id_vars, _ = measure_sep_structures(sep_points, sep_rec_cpca_id_data, dist_measure, k_neigh)

                        sep_transf_data = [transf_data[i:i + num_points_per_line] for i in range(0, len(transf_data), num_points_per_line)]
                        sep_transf_cpca_real_pca_data = [transf_cpca_real_pca_data[i:i + num_points_per_line] for i in range(0, len(transf_cpca_real_pca_data), num_points_per_line)]
                        sep_transf_cpca_real_low_data = [transf_cpca_real_low_data[i:i + num_points_per_line] for i in range(0, len(transf_cpca_real_low_data), num_points_per_line)]
                        sep_transf_cpca_imag_data = [transf_cpca_imag_data[i:i + num_points_per_line] for i in range(0, len(transf_cpca_imag_data), num_points_per_line)]
                        sep_transf_cpca_id_data = [transf_id_data[i:i + num_points_per_line] for i in range(0, len(transf_id_data), num_points_per_line)]

                        transf_exp_values, transf_vars, _ = measure_sep_structures(sep_points, sep_transf_data, dist_measure, k_neigh)
                        transf_exp_cpca_real_pca_values, transf_cpca_real_pca_vars, _ = measure_sep_structures(sep_points, sep_transf_cpca_real_pca_data, dist_measure, k_neigh)
                        transf_exp_cpca_low_values, transf_cpca_low_vars, _ = measure_sep_structures(sep_points, sep_transf_cpca_real_low_data, dist_measure, k_neigh)
                        transf_exp_cpca_imag_values, transf_cpca_imag_vars, _ = measure_sep_structures(sep_points, sep_transf_cpca_imag_data, dist_measure, k_neigh)
                        transf_exp_id_values, transf_id_vars, _ = measure_sep_structures(sep_points, sep_transf_cpca_id_data, dist_measure, k_neigh)

                # Append
                list_pca_means.append(reconstr_exp_values[-1])
                list_pca_vars.append(reconstr_vars)
                list_cpca_real_pca_means.append(reconstr_exp_cpca_real_pca_values[-1])
                list_cpca_real_pca_vars.append(reconstr_cpca_real_pca_vars[-1])

                list_cpca_real_low_means.append(reconstr_exp_cpca_low_values[-1])
                list_cpca_real_low_vars.append(reconstr_cpca_low_vars[-1])

                list_cpca_imag_means.append(reconstr_exp_cpca_imag_values[-1])
                list_cpca_imag_vars.append(reconstr_cpca_imag_vars[-1])
                
                list_id_means.append(reconstr_exp_id_values[-1])
                list_id_vars.append(reconstr_id_vars[-1])

                if dist_measure != "dtw":
                    list_transf_pca_means.append(transf_exp_values[-1])
                    list_transf_pca_vars.append(transf_vars[-1])
                    
                    list_transf_cpca_real_pca_means.append(transf_exp_cpca_real_pca_values)
                    list_transf_cpca_real_pca_vars.append(transf_cpca_real_pca_vars)

                    list_transf_cpca_real_low_means.append(transf_exp_cpca_low_values)
                    list_transf_cpca_real_low_vars.append(transf_cpca_low_vars)

                    list_transf_cpca_imag_means.append(transf_exp_cpca_imag_values)
                    list_transf_cpca_imag_vars.append(transf_cpca_imag_vars)
                    
                    list_transf_id_means.append(transf_exp_id_values)
                    list_transf_id_vars.append(transf_id_vars)

                
            logging.info("Finished computing")
            print(list_cpca_real_low_means)
            print(list_pca_means)
            pickle_cpca_results_avg_dev_vs_dyn_dim("./pickled_results/cpca_avg_dev_vs_dyn_low_dims.pickle", [s, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end, dim_list], [(list_pca_means, list_pca_vars), (list_cpca_real_pca_means, list_cpca_real_pca_vars), (list_cpca_real_low_means, list_cpca_real_low_vars), (list_cpca_imag_means, list_cpca_imag_vars), (list_id_means, list_id_vars), (list_transf_pca_means, list_transf_pca_vars), (list_transf_cpca_real_pca_means, list_transf_cpca_real_pca_vars), (list_transf_cpca_real_low_means, list_transf_cpca_real_low_vars), (list_transf_cpca_imag_means, list_transf_cpca_imag_vars), (list_transf_id_means, list_transf_id_vars)])
     
            
def plot_cpca_mean_of_multiple_runs_avg_dev_vs_dyn_low_dim(start_dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, num_runs, variance = False, cut_k = None):
    dims_list = generate_integer_list_dyn_low(start_dimensions, num_lines, order)
    
    seed_list = list(range(num_runs))
    
    parameters = [start_dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, dims_list]
    
    result_list = find_in_pickle_all_without_seed("./pickled_results/cpca_avg_dev_vs_dyn_low_dims.pickle", parameters)
    
    slurm_result_list = find_in_pickle_all_without_seed("./pickled_results/slurm/cpca_avg_dev_vs_dyn_low_dims.pickle", parameters)
    
    assert result_list is not None or slurm_result_list is not None, "Error: There are no pickled results for pca, vector pca and random-projections"
    try:
        if len(slurm_result_list) != 0:
            result_list = slurm_result_list
    except:
        pass
    result_list = [entry for entry in result_list if entry["parameters"][0] in seed_list]
    
    assert len(result_list) == num_runs

    mean_list_pca_means = []
    mean_list_pca_vars = []

    mean_list_cpca_real_pca_means = []
    mean_list_cpca_real_pca_vars = []

    mean_list_cpca_real_low_means = []
    mean_list_cpca_real_low_vars = []

    mean_list_cpca_imag_means = []
    mean_list_cpca_imag_vars = []
    
    mean_list_id_means = []
    mean_list_id_vars = []
    
    mean_list_transf_pca_means = []
    mean_list_transf_pca_vars = []
    
    mean_list_transf_cpca_real_pca_means = []
    mean_list_transf_cpca_real_pca_vars = []

    mean_list_transf_cpca_real_low_means = []
    mean_list_transf_cpca_real_low_vars = []

    mean_list_transf_cpca_imag_means = []
    mean_list_transf_cpca_imag_vars = []
    
    mean_list_transf_id_means = []
    mean_list_transf_id_vars = []
    
    for res in result_list:
        mean_list_pca_means.append(res["results"][0][0])
        mean_list_pca_vars.append(res["results"][0][1])
        mean_list_cpca_real_pca_means.append(res["results"][1][0])
        mean_list_cpca_real_pca_vars.append(res["results"][1][1])
        mean_list_cpca_real_low_means.append(res["results"][2][0])
        mean_list_cpca_real_low_vars.append(res["results"][2][1])
        mean_list_cpca_imag_means.append(res["results"][3][0])
        mean_list_cpca_imag_vars.append(res["results"][3][1])
        mean_list_id_means.append(res["results"][4][0])
        mean_list_id_vars.append(res["results"][4][1])
        
        mean_list_transf_pca_means.append(res["results"][5][0])
        mean_list_transf_pca_vars.append(res["results"][5][1])
        mean_list_transf_cpca_real_pca_means.append(res["results"][6][0])
        mean_list_transf_cpca_real_pca_vars.append(res["results"][6][1])
        mean_list_transf_cpca_real_low_means.append(res["results"][7][0])
        mean_list_transf_cpca_real_low_vars.append(res["results"][7][1])
        mean_list_transf_cpca_imag_means.append(res["results"][8][0])
        mean_list_transf_cpca_imag_vars.append(res["results"][8][1])
        mean_list_transf_id_means.append(res["results"][9][0])
        mean_list_transf_id_vars.append(res["results"][9][1])
    mean_pca_mean_list = np.mean(mean_list_pca_means, axis=0)[:cut_k]
    mean_pca_var_list = flatten(np.mean(mean_list_pca_vars, axis=0)[:cut_k])
    mean_list_cpca_real_pca_mean_list = np.mean(mean_list_cpca_real_pca_means, axis=0)[:cut_k]
    mean_list_cpca_real_pca_var_list = np.mean(mean_list_cpca_real_pca_vars, axis=0)[:cut_k]
    mean_list_cpca_real_low_mean_list = np.mean(mean_list_cpca_real_low_means, axis=0)[:cut_k]
    mean_list_cpca_real_low_var_list = np.mean(mean_list_cpca_real_low_vars, axis=0)[:cut_k]
    mean_list_cpca_imag_mean_list = np.mean(mean_list_cpca_imag_means, axis=0)[:cut_k]
    mean_list_cpca_imag_var_list = np.mean(mean_list_cpca_imag_vars, axis=0)[:cut_k]
    mean_list_id_mean_list = np.mean(mean_list_id_means, axis=0)[:cut_k]
    mean_list_id_var_list = np.mean(mean_list_id_vars, axis=0)[:cut_k]
    
    transf_mean_pca_mean_list = np.mean(mean_list_transf_pca_means, axis=0)[:cut_k]
    transf_mean_pca_var_list = np.mean(mean_list_transf_pca_vars, axis=0)[:cut_k]
    transf_mean_list_cpca_real_pca_mean_list = flatten(np.mean(mean_list_transf_cpca_real_pca_means, axis=0)[:cut_k])
    transf_mean_list_cpca_real_pca_var_list = flatten(np.mean(mean_list_transf_cpca_real_pca_vars, axis=0)[:cut_k])
    transf_mean_list_cpca_real_low_mean_list = flatten(np.mean(mean_list_transf_cpca_real_low_means, axis=0)[:cut_k])
    transf_mean_list_cpca_real_low_var_list = flatten(np.mean(mean_list_transf_cpca_real_low_vars, axis=0)[:cut_k])
    transf_mean_list_cpca_imag_mean_list = flatten(np.mean(mean_list_transf_cpca_imag_means, axis=0)[:cut_k])
    transf_mean_list_cpca_imag_var_list = flatten(np.mean(mean_list_transf_cpca_imag_vars, axis=0)[:cut_k])
    transf_mean_list_id_mean_list = flatten(np.mean(mean_list_transf_id_means, axis=0)[:cut_k])
    transf_mean_list_id_var_list = flatten(np.mean(mean_list_transf_id_vars, axis=0)[:cut_k])
    
    if dist_measure == "dtw":
        fig, ax1 = plt.subplots(figsize=(14, 10))
    else:
        fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 10))
    
    dims_list = dims_list[:cut_k]

    # Plot for random projection for reconstructed data
    upper_var = np.array(mean_list_cpca_real_pca_mean_list) + np.array(mean_list_cpca_real_pca_var_list)
    lower_var = np.array(mean_list_cpca_real_pca_mean_list) - np.array(mean_list_cpca_real_pca_var_list)

    if sep_measure:
        ax1.plot(dims_list, mean_list_cpca_real_pca_mean_list, label="CPCA-real: scaled pca", color = "r", linestyle="dashdot", linewidth=3)
        
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "r")
    else:
        ax1.plot(dims_list, mean_list_cpca_real_pca_mean_list, label="CPCA-real: scaled pca", color = "r", linestyle="dashdot", linewidth=3)
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "r", linewidth = 2)
        
    # Plot for Vector PCA for reconstructed data
    upper_var = np.array(mean_list_cpca_real_low_mean_list) + np.array(mean_list_cpca_real_low_var_list)
    lower_var = np.array(mean_list_cpca_real_low_mean_list) - np.array(mean_list_cpca_real_low_var_list)

    if sep_measure:
        ax1.plot(dims_list, mean_list_cpca_real_low_mean_list, label="CPCA-real: scaled low imag", color = "m", linewidth=3, linestyle = "dotted")
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "m")
    else: 
        ax1.plot(dims_list, mean_list_cpca_real_low_mean_list, label="CPCA-real: scaled low imag", color = "m", linewidth=3, linestyle = "dotted")
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "m", linewidth = 2)
            
     # Plot for Vector PCA for reconstructed data
    upper_var = np.array(mean_list_cpca_imag_mean_list) + np.array(mean_list_cpca_imag_var_list)
    lower_var = np.array(mean_list_cpca_imag_mean_list) - np.array(mean_list_cpca_imag_var_list)

    if sep_measure:
        ax1.plot(dims_list, mean_list_cpca_imag_mean_list, label="CPCA-imag", color = "orange", linewidth=3, linestyle = (0, (3, 5, 1, 5, 1, 5)))
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "orange")
    else: 
        ax1.plot(dims_list, mean_list_cpca_imag_mean_list, label="CPCA-imag", color = "orange", linewidth=3, linestyle = (0, (3, 5, 1, 5, 1, 5)))
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "orange", linewidth = 2)
    
     # Plot for Vector PCA for reconstructed data
    upper_var = np.array(mean_list_id_mean_list) + np.array(mean_list_id_var_list)
    lower_var = np.array(mean_list_id_mean_list) - np.array(mean_list_id_var_list)

    if sep_measure:
        ax1.plot(dims_list, mean_list_id_mean_list, label="Id PCA", color = "b", linewidth=3, linestyle= (0, (3, 1, 1, 1, 1, 1)))
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "b")
    else: 
        ax1.plot(dims_list, mean_list_id_mean_list, label="Id PCA", color = "b", linewidth=3, linestyle = (0, (3, 1, 1, 1, 1, 1)))
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "b", linewidth = 2)
        
    # Plot for PCA for reconstructed data
    upper_var = np.array(mean_pca_mean_list) + np.array(mean_pca_var_list)
    lower_var = np.array(mean_pca_mean_list) - np.array(mean_pca_var_list)

    if sep_measure:
        ax1.plot(dims_list, mean_pca_mean_list, label="PCA", color = "#007a33", linewidth=3, linestyle = "dashed")
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "#007a33")
    else: 
        ax1.plot(dims_list, mean_pca_mean_list, label="PCA", color = "#007a33", linewidth=3, linestyle = "dashed")
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "#007a33", linewidth = 2)
            
    if dist_measure != "dtw":  
        # Plot for tsne for transformed data
        upper_var = np.array(transf_mean_list_cpca_real_pca_mean_list) + np.array(transf_mean_list_cpca_real_pca_var_list)
        lower_var = np.array(transf_mean_list_cpca_real_pca_mean_list) - np.array(transf_mean_list_cpca_real_pca_var_list)

        if sep_measure:
            ax2.plot(dims_list, transf_mean_list_cpca_real_pca_mean_list, label="CPCA-real: scaled pca", linestyle="dashdot", color="r", linewidth=3)
            if variance:
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color="r",)
        else: 
            ax2.plot(dims_list, transf_mean_list_cpca_real_pca_mean_list, label="CPCA-real: scaled pca", linestyle="dashdot", color="r", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color="r")
        
        # Plot for umap for transformed data
        upper_var = np.array(transf_mean_list_cpca_real_low_mean_list) + np.array(transf_mean_list_cpca_real_low_var_list)
        lower_var = np.array(transf_mean_list_cpca_real_low_mean_list) - np.array(transf_mean_list_cpca_real_low_var_list)

        if sep_measure:
            ax2.plot(dims_list, transf_mean_list_cpca_real_low_mean_list, label="CPCA-real: scaled low imag", linestyle = "dotted", color="m", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color="m")
        else: 
            ax2.plot(dims_list, transf_mean_list_cpca_real_low_mean_list, label="CPCA-real: scaled low imag", linestyle = "dotted", color="m", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color="m")
            
        # Plot for random projection for transformed data
        upper_var = np.array(transf_mean_list_cpca_imag_mean_list) + np.array(transf_mean_list_cpca_imag_var_list)
        lower_var = np.array(transf_mean_list_cpca_imag_mean_list) - np.array(transf_mean_list_cpca_imag_var_list)

        if sep_measure:
            ax2.plot(dims_list, transf_mean_list_cpca_imag_mean_list, label="CPCA-imag", color = "orange", linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "orange")
        else: 
            ax2.plot(dims_list, transf_mean_list_cpca_imag_mean_list, label="CPCA-imag", color = "orange", linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "orange", linewidth = 2)
        
        # Plot for PCA for transformed data
        upper_var = np.array(transf_mean_list_id_mean_list) + np.array(transf_mean_list_id_var_list)
        lower_var = np.array(transf_mean_list_id_mean_list) - np.array(transf_mean_list_id_var_list)

        if sep_measure:
            ax2.plot(dims_list, transf_mean_list_id_mean_list, label="Id PCA", color = "b", linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "b")
        else: 
            ax2.plot(dims_list, transf_mean_list_id_mean_list, label="Id PCA", color = "b", linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "b", linewidth = 2)
        
        # Plot for lle for transformed data
        upper_var = np.array(transf_mean_pca_mean_list) + np.array(transf_mean_pca_var_list)
        lower_var = np.array(transf_mean_pca_mean_list) - np.array(transf_mean_pca_var_list)

        if sep_measure:
            ax2.plot(dims_list, transf_mean_pca_mean_list, label="PCA", color="#007a33", linewidth=3, linestyle = "dashed")
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color="#007a33")
        else: 
            ax2.plot(dims_list, transf_mean_pca_mean_list, label="PCA", color="#007a33", linewidth=3, linestyle = "dashed")
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color="#007a33")
        

    if dist_measure != "dtw":
        fontsize = plot_dict["lineplots"]["fontsize"]
        fontdist = plot_dict["lineplots"]["font_dist"]
        fontlegend = plot_dict["lineplots"]["font_legend"]
        if cut_k is None:
            x_steps = 5
        else:
            x_steps = 2
    else:
        fontsize = plot_dict["lineplots"]["dtw_fontsize"]
        fontdist = plot_dict["lineplots"]["dtw_fontdist"]
        fontlegend = plot_dict["lineplots"]["dtw_font_legend"]

    ax1.set_xlabel("Dimension " + str(start_dimensions) + " reduced to", fontsize = fontsize - fontdist)
    ax1.set_ylabel("Avg. dev. from \n orig. data", fontsize = fontsize - fontdist)
    ax1.set_title("Results for Reconstruction of " + str(num_runs) + " runs", fontsize = fontsize)
    ax1.legend(fontsize = fontlegend, bbox_to_anchor=(1.06, 1))
    ax1.tick_params(direction='out', labelsize=fontsize -fontdist)
    ax1.set_axisbelow(True)
    
    ax1.set_xticks([d for d in dims_list if d % x_steps == 0])
    range_max = max(flatten([mean_pca_mean_list, mean_list_cpca_real_pca_mean_list, mean_list_cpca_real_low_mean_list, mean_list_cpca_imag_mean_list, mean_list_id_mean_list]))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.yaxis.get_offset_text().set_fontsize(fontsize - fontdist)
    next_higher = pow(10, len(str(round(range_max))))
    if range_max < int(next_higher / 4):
            next_higher = int(next_higher / 4)
            if order == "zigzag" and dist_measure == "multiple_scalar_product":
                ax1.ticklabel_format(style='sci', axis='y', scilimits=(4, 4), useOffset = True)
    parts = 5
    ax1.set_yticks(range(0, next_higher, int(next_higher / parts)))
    if range_max < next_higher - int(next_higher / 2):
            next_higher = next_higher - int(next_higher / 2)
    if dist_measure not in ["multiple_scalar_product", "scalar_product"]:
        variance_max = max(flatten([mean_pca_var_list, mean_pca_var_list, mean_list_cpca_real_pca_var_list, mean_list_cpca_real_low_var_list, mean_list_cpca_imag_var_list, mean_list_id_var_list]))
        range_max += variance_max
        
    elif order in ["sep_lines"]:
        if cut_k is not None:
            ax1.ticklabel_format(style='sci', axis='y', scilimits=(2, 2), useOffset = True)
            ax1.set_ylim(700)
            ax1.set_yticks(range(700, next_higher - 50, 50))
            
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.grid(axis='y', linestyle="dashed", color="silver", linewidth=1.5)
    
    if dist_measure != "dtw":
        ax2.set_xlabel("Dimension " + str(start_dimensions) + " reduced to", fontsize = fontsize - fontdist)
        ax2.set_ylabel("Avg. dev. from \n orig. data", fontsize = fontsize - fontdist)
        ax2.set_title("Results for Transformations of " + str(num_runs) + " runs", fontsize = fontsize)
        ax2.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, fontsize = fontlegend)
        ax2.tick_params(direction='out', labelsize=fontsize - fontdist)
        ax2.set_axisbelow(True)
        
        ax2.set_xticks([d for d in dims_list if d % x_steps == 0])
        range_max = max(flatten([transf_mean_pca_mean_list, transf_mean_list_cpca_real_pca_mean_list, transf_mean_list_cpca_real_low_mean_list, transf_mean_list_cpca_imag_mean_list, transf_mean_list_id_mean_list]))
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax2.yaxis.get_offset_text().set_fontsize(fontsize - fontdist)
        next_higher = pow(10, len(str(round(range_max))))
        
        if range_max < int(next_higher / 4):
            next_higher = int(next_higher / 4)
            if order == "zigzag" and dist_measure == "multiple_scalar_product":
                ax2.ticklabel_format(style='sci', axis='y', scilimits=(4, 4), useOffset = True)
        parts = 5
        ax2.set_yticks(range(0, next_higher, int(next_higher / parts)))
        if range_max < next_higher - int(next_higher / 2):
            next_higher = next_higher - int(next_higher / 2)
        if dist_measure not in ["multiple_scalar_product", "scalar_product"]:
            variance_max = max(flatten([transf_mean_pca_var_list, transf_mean_list_cpca_real_pca_var_list, transf_mean_list_cpca_real_low_var_list, transf_mean_list_cpca_imag_var_list, transf_mean_list_id_var_list]))
            range_max += variance_max
    
        elif order in ["sep_lines"]:
            if cut_k is not None:
                ax2.ticklabel_format(style='sci', axis='y', scilimits=(2, 2), useOffset = True)
                ax2.set_ylim(700)
                ax2.set_yticks(range(700, next_higher - 50, 50))
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.grid(axis='y', linestyle="dashed", color="silver", linewidth=1.5)
        
    # plt.suptitle("Order " + order + " measure " + str(k) + " neighbours for " + str(num_lines) + " lines a " + str(num_points) + " and jitter " + str(jitter_bound))
    if dist_measure == "dtw":
        fig.tight_layout()
    else:
        fig.tight_layout(h_pad = 4)
        # fig.set_figwidth(14)
        # fig.set_figheight(8)
        
    if cut_k is None:
        plt.savefig("./plots/generated/multiple_runs/cpca/" + str(order) + "/avg_dev_vs_dyn_low/" + str(num_lines) + "lines_" + str(num_points) + "points_" + str(k) + "neighbours_" + str(dist_measure) + ".pdf", bbox_inches="tight")
    else:
        plt.savefig("./plots/generated/multiple_runs/cpca/" + str(order) + "/avg_dev_vs_dyn_low/" + str(num_lines) + "lines_" + str(num_points) + "points_" + str(k) + "neighbours_" + str(dist_measure) + "_cutk" + str(cut_k) + ".pdf", bbox_inches="tight")
    # ax1.set_yscale("log")
    plt.show()


def cpca_multiple_runs_neighb_error(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, target_dimensions, jitter_bound, num_points, num_lines, max_k, num_runs, scaling_choice = "vector_pca"):
    assert dist_measure != "scalar_product", "Error: Please use for this function the measure multiple_scalar_product"
    
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    
    slurm_already_computed = find_in_pickle_all_without_seed("./pickled_results/slurm/cpca_neighb_error.pickle", [dimensions, target_dimensions, order, bounds, num_lines, num_points, dist_measure, max_k, sep_measure, jitter_bound, parallel_start_end, num_runs, scaling_choice])
    
    already_computed = find_in_pickle_all_without_seed("./pickled_results/cpca_neighb_error.pickle", [dimensions, target_dimensions, order, bounds, num_lines, num_points, dist_measure, max_k, sep_measure, jitter_bound, parallel_start_end, num_runs, scaling_choice])
    
    
    try:
        if len(slurm_already_computed) != 0:
            already_computed = slurm_already_computed
    except:
        pass
    
    seed_list = list(range(num_runs))
    if already_computed is not None:
        already_seeds = [r["parameters"][0] for r in already_computed]
    else:
        already_seeds = []
    seed_list = [s for s in seed_list if s not in flatten([already_seeds])]
    
    for s in seed_list:
        logging.info("Start seed " + str(s))
        random.seed(s)
        np.random.seed(s)
        
        sep_points, separated = generate_correct_ordered_data(order, num_points, num_lines, dimensions, jitter_bound, bounds, parallel_start_end)
        
        if separated:
            data = flatten(sep_points)
        else:
            data = sep_points
        
        if s not in already_seeds:
            logging.info("Not found")
             # Compute the transformed and reconstructed data for original, complex and ID data
            reconstr_data, transf_data = pca_reconstruction(data, target_dimensions, "complex_pca", None, None)
            
            complex_data = add_complex_vector(sep_points, separated)

            # Transform and reconstruct data with cpca and scale to original pca
            logging.info("Get vector data")
            reconstr_cpca_real_pca_data, transf_cpca_real_pca_data = complex_PCA.pca_reconstruction(complex_data, target_dimensions, "complex_pca_svd_complex", "like_original_pca", None)
            
            # Transform and reconstruct data with cpca and scale to lowest imaginary part
            logging.info("Get vector data")
            reconstr_cpca_real_low_data, transf_cpca_real_low_data = complex_PCA.pca_reconstruction(complex_data, target_dimensions, "complex_pca_svd_complex", "low_imag", None)
            
            # Transform and reconstruct data with cpca and scale to lowest imaginary part
            logging.info("Get vector data")
            reconstr_cpca_imag_data, transf_cpca_imag_data = complex_PCA.pca_reconstruction(complex_data, target_dimensions, "complex_pca_svd_complex_imag_pc", "low_imag", None)
            
            # Add Ids and reconstruct and transform data
            if separated:
                id_data = flatten(add_id(sep_points, separated))
            else:
                id_data = add_id(sep_points, separated)
            reconstr_id_data, transf_id_data = pca_reconstruction(id_data, target_dimensions + 1, "complex_pca", None, None)
            reconstr_id_data = [elem[:-1] for elem in reconstr_id_data]
            transf_id_data = [elem[:-1] for elem in transf_id_data]

            # Compute mean and std for original data, complex transformed data, ID data
            if not sep_measure:
                reconstr_exp_values, reconstr_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_data, dist_measure, max_k)
                    
                reconstr_exp_cpca_real_pca_values, reconstr_cpca_real_pca_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_cpca_real_pca_data, dist_measure, max_k)
                reconstr_exp_cpca_low_values, reconstr_cpca_low_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_cpca_real_low_data, dist_measure, max_k)
                reconstr_exp_cpca_imag_values, reconstr_cpca_imag_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_cpca_imag_data, dist_measure, max_k)
                reconstr_exp_id_values, reconstr_id_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_id_data, dist_measure, max_k)
            
                transf_exp_values, transf_vars, _ = get_mu_var_by_neighbor_num(data, transf_data, dist_measure, max_k)
                transf_exp_cpca_real_pca_values, transf_cpca_real_pca_vars, _ = get_mu_var_by_neighbor_num(data, transf_cpca_real_pca_data, dist_measure, max_k)
                transf_exp_cpca_low_values, transf_cpca_low_vars, _ = get_mu_var_by_neighbor_num(data, transf_cpca_real_low_data, dist_measure, max_k)
                transf_exp_cpca_imag_values, transf_cpca_imag_vars, _ = get_mu_var_by_neighbor_num(data, transf_cpca_imag_data, dist_measure, max_k)
                transf_exp_id_values, transf_id_vars, _ = get_mu_var_by_neighbor_num(data, transf_id_data, dist_measure, max_k)
                
            elif sep_measure:
                sep_rec_data = [reconstr_data[i:i + num_points] for i in range(0, len(reconstr_data), num_points)]
                sep_rec_cpca_real_pca_data = [reconstr_cpca_real_pca_data[i:i + num_points] for i in range(0, len(reconstr_cpca_real_pca_data), num_points)]
                sep_rec_cpca_real_low_data = [reconstr_cpca_real_low_data[i:i + num_points] for i in range(0, len(reconstr_cpca_real_low_data), num_points)]
                sep_rec_cpca_imag_data = [reconstr_cpca_imag_data[i:i + num_points] for i in range(0, len(reconstr_cpca_imag_data), num_points)]
                sep_rec_cpca_id_data = [reconstr_id_data[i:i + num_points] for i in range(0, len(reconstr_id_data), num_points)]
                
                reconstr_exp_values, reconstr_vars, _ = measure_sep_structures(sep_points, sep_rec_data, dist_measure, max_k)
                reconstr_exp_cpca_real_pca_values, reconstr_cpca_real_pca_vars, _ = measure_sep_structures(sep_points, sep_rec_cpca_real_pca_data, dist_measure, max_k)
                reconstr_exp_cpca_low_values, reconstr_cpca_low_vars, _ = measure_sep_structures(sep_points, sep_rec_cpca_real_low_data, dist_measure, max_k)
                reconstr_exp_cpca_imag_values, reconstr_cpca_imag_vars, _ = measure_sep_structures(sep_points, sep_rec_cpca_imag_data, dist_measure, max_k)
                reconstr_exp_id_values, reconstr_id_vars, _ = measure_sep_structures(sep_points, sep_rec_cpca_id_data, dist_measure, max_k)

                sep_transf_data = [transf_data[i:i + num_points] for i in range(0, len(transf_data), num_points)]
                sep_transf_cpca_real_pca_data = [transf_cpca_real_pca_data[i:i + num_points] for i in range(0, len(transf_cpca_real_pca_data), num_points)]
                sep_transf_cpca_real_low_data = [transf_cpca_real_low_data[i:i + num_points] for i in range(0, len(transf_cpca_real_low_data), num_points)]
                sep_transf_cpca_imag_data = [transf_cpca_imag_data[i:i + num_points] for i in range(0, len(transf_cpca_imag_data), num_points)]
                sep_transf_cpca_id_data = [transf_id_data[i:i + num_points] for i in range(0, len(transf_id_data), num_points)]

                transf_exp_values, transf_vars, _ = measure_sep_structures(sep_points, sep_transf_data, dist_measure, max_k)
                transf_exp_cpca_real_pca_values, transf_cpca_real_pca_vars, _ = measure_sep_structures(sep_points, sep_transf_cpca_real_pca_data, dist_measure, max_k)
                transf_exp_cpca_low_values, transf_cpca_low_vars, _ = measure_sep_structures(sep_points, sep_transf_cpca_real_low_data, dist_measure, max_k)
                transf_exp_cpca_imag_values, transf_cpca_imag_vars, _ = measure_sep_structures(sep_points, sep_transf_cpca_imag_data, dist_measure, max_k)
                transf_exp_id_values, transf_id_vars, _ = measure_sep_structures(sep_points, sep_transf_cpca_id_data, dist_measure, max_k)
                
            pickle_cpca_results_avg_dev_vs_dyn_dim("./pickled_results/cpca_neighb_error.pickle", [s, dimensions, target_dimensions, order, bounds, num_lines, num_points, dist_measure, max_k, sep_measure, jitter_bound, parallel_start_end, num_runs, scaling_choice], [(reconstr_exp_values, reconstr_vars), (reconstr_exp_cpca_real_pca_values, reconstr_cpca_real_pca_vars), (reconstr_exp_cpca_low_values, reconstr_cpca_low_vars), (reconstr_exp_cpca_imag_values, reconstr_cpca_imag_vars), (reconstr_exp_id_values, reconstr_id_vars), (transf_exp_values, transf_vars), (transf_exp_cpca_real_pca_values, transf_cpca_real_pca_vars), (transf_exp_cpca_low_values, transf_cpca_low_vars), (transf_exp_cpca_imag_values, transf_cpca_imag_vars), (transf_exp_id_values, transf_id_vars)])
            
def cpca_plot_multiple_runs_neighb_error(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, target_dimensions, jitter_bound, num_points, num_lines, max_k, num_runs, variance, cut_k = None, scaling_choice = "vector_pca"):
    assert dist_measure != "dtw", "Error: DTW is not suitable"
    assert dist_measure != "scalar_product", "Error: Please use for this function the measure multiple_scalar_product"
    
    seed_list = list(range(num_runs))
    
    parameters = [dimensions, target_dimensions, order, bounds, num_lines, num_points, dist_measure, max_k, sep_measure, jitter_bound, parallel_start_end, num_runs, scaling_choice]
    result_list = find_in_pickle_all_without_seed("./pickled_results/cpca_neighb_error.pickle", parameters)
    slurm_result_list = find_in_pickle_all_without_seed("./pickled_results/slurm/cpca_neighb_error.pickle", parameters)
    
    if len(result_list) != num_runs:
        result_list = slurm_result_list
    
    assert result_list is not None, "Error: There are no pickled results for pca, vector pca and random-projections"
    
    result_list = [entry for entry in result_list if entry["parameters"][0] in seed_list]
    assert len(result_list) == num_runs
    
    mean_list_pca_means = []
    mean_list_pca_vars = []

    mean_list_cpca_real_pca_means = []
    mean_list_cpca_real_pca_vars = []

    mean_list_cpca_real_low_means = []
    mean_list_cpca_real_low_vars = []

    mean_list_cpca_imag_means = []
    mean_list_cpca_imag_vars = []
    
    mean_list_id_means = []
    mean_list_id_vars = []
    
    mean_list_transf_pca_means = []
    mean_list_transf_pca_vars = []
    
    mean_list_transf_cpca_real_pca_means = []
    mean_list_transf_cpca_real_pca_vars = []

    mean_list_transf_cpca_real_low_means = []
    mean_list_transf_cpca_real_low_vars = []

    mean_list_transf_cpca_imag_means = []
    mean_list_transf_cpca_imag_vars = []
    
    mean_list_transf_id_means = []
    mean_list_transf_id_vars = []
    
    for res in result_list:
        mean_list_pca_means.append(res["results"][0][0])
        mean_list_pca_vars.append(res["results"][0][1])
        mean_list_cpca_real_pca_means.append(res["results"][1][0])
        mean_list_cpca_real_pca_vars.append(res["results"][1][1])
        mean_list_cpca_real_low_means.append(res["results"][2][0])
        mean_list_cpca_real_low_vars.append(res["results"][2][1])
        mean_list_cpca_imag_means.append(res["results"][3][0])
        mean_list_cpca_imag_vars.append(res["results"][3][1])
        mean_list_id_means.append(res["results"][4][0])
        mean_list_id_vars.append(res["results"][4][1])
        
        mean_list_transf_pca_means.append(res["results"][5][0])
        mean_list_transf_pca_vars.append(res["results"][5][1])
        mean_list_transf_cpca_real_pca_means.append(res["results"][6][0])
        mean_list_transf_cpca_real_pca_vars.append(res["results"][6][1])
        mean_list_transf_cpca_real_low_means.append(res["results"][7][0])
        mean_list_transf_cpca_real_low_vars.append(res["results"][7][1])
        mean_list_transf_cpca_imag_means.append(res["results"][8][0])
        mean_list_transf_cpca_imag_vars.append(res["results"][8][1])
        mean_list_transf_id_means.append(res["results"][9][0])
        mean_list_transf_id_vars.append(res["results"][9][1])
    
    mean_pca_mean_list = np.mean(mean_list_pca_means, axis=0)[:cut_k]
    mean_pca_var_list = np.mean(mean_list_pca_vars, axis=0)[:cut_k]
    mean_list_cpca_real_pca_mean_list = np.mean(mean_list_cpca_real_pca_means, axis=0)[:cut_k]
    mean_list_cpca_real_pca_var_list = np.mean(mean_list_cpca_real_pca_vars, axis=0)[:cut_k]
    mean_list_cpca_real_low_mean_list = np.mean(mean_list_cpca_real_low_means, axis=0)[:cut_k]
    mean_list_cpca_real_low_var_list = np.mean(mean_list_cpca_real_low_vars, axis=0)[:cut_k]
    mean_list_cpca_imag_mean_list = np.mean(mean_list_cpca_imag_means, axis=0)[:cut_k]
    mean_list_cpca_imag_var_list = np.mean(mean_list_cpca_imag_vars, axis=0)[:cut_k]
    mean_list_id_mean_list = np.mean(mean_list_id_means, axis=0)[:cut_k]
    mean_list_id_var_list = np.mean(mean_list_id_vars, axis=0)[:cut_k]
    
    transf_mean_pca_mean_list = np.mean(mean_list_transf_pca_means, axis=0)[:cut_k]
    transf_mean_pca_var_list = np.mean(mean_list_transf_pca_vars, axis=0)[:cut_k]
    transf_mean_list_cpca_real_pca_mean_list = np.mean(mean_list_transf_cpca_real_pca_means, axis=0)[:cut_k]
    transf_mean_list_cpca_real_pca_var_list = np.mean(mean_list_transf_cpca_real_pca_vars, axis=0)[:cut_k]
    transf_mean_list_cpca_real_low_mean_list = np.mean(mean_list_transf_cpca_real_low_means, axis=0)[:cut_k]
    transf_mean_list_cpca_real_low_var_list = np.mean(mean_list_transf_cpca_real_low_vars, axis=0)[:cut_k]
    transf_mean_list_cpca_imag_mean_list = np.mean(mean_list_transf_cpca_imag_means, axis=0)[:cut_k]
    transf_mean_list_cpca_imag_var_list = np.mean(mean_list_transf_cpca_imag_vars, axis=0)[:cut_k]
    transf_mean_list_id_mean_list = np.mean(mean_list_transf_id_means, axis=0)[:cut_k]
    transf_mean_list_id_var_list = np.mean(mean_list_transf_id_vars, axis=0)[:cut_k]

    fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 10))
    dims_list = range(max_k)

    # Plot for random projection for reconstructed data
    upper_var = np.array(mean_list_cpca_real_pca_mean_list) + np.array(mean_list_cpca_real_pca_var_list)
    lower_var = np.array(mean_list_cpca_real_pca_mean_list) - np.array(mean_list_cpca_real_pca_var_list)

    if sep_measure:
        ax1.plot(dims_list, mean_list_cpca_real_pca_mean_list, label="CPCA-real: scaled pca", color = "r", linestyle="dashdot", linewidth=3)
        
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "r")
    else: 
        ax1.plot(dims_list, mean_list_cpca_real_pca_mean_list, label="CPCA-real: scaled pca", color = "r", linestyle="dashdot", linewidth=3)
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "r", linewidth = 2)
        
    # Plot for Vector PCA for reconstructed data
    upper_var = np.array(mean_list_cpca_real_low_mean_list) + np.array(mean_list_cpca_real_low_var_list)
    lower_var = np.array(mean_list_cpca_real_low_mean_list) - np.array(mean_list_cpca_real_low_var_list)

    if sep_measure:
        ax1.plot(dims_list, mean_list_cpca_real_low_mean_list, label="CPCA-real: scaled low imag", color = "m", linewidth=3, linestyle = "dotted")
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "m")
    else: 
        ax1.plot(dims_list, mean_list_cpca_real_low_mean_list, label="CPCA-real: scaled low imag", color = "m", linewidth=3, linestyle = "dotted")
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "m", linewidth = 2)
            
     # Plot for Vector PCA for reconstructed data
    upper_var = np.array(mean_list_cpca_imag_mean_list) + np.array(mean_list_cpca_imag_var_list)
    lower_var = np.array(mean_list_cpca_imag_mean_list) - np.array(mean_list_cpca_imag_var_list)

    if sep_measure:
        ax1.plot(dims_list, mean_list_cpca_imag_mean_list, label="CPCA-imag", color = "orange", linewidth=3, linestyle = (0, (3, 5, 1, 5, 1, 5)))
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "orange")
    else: 
        ax1.plot(dims_list, mean_list_cpca_imag_mean_list, label="CPCA-imag", color = "orange", linewidth=3, linestyle = (0, (3, 5, 1, 5, 1, 5)))
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "orange", linewidth = 2)
    
     # Plot for Vector PCA for reconstructed data
    upper_var = np.array(mean_list_id_mean_list) + np.array(mean_list_id_var_list)
    lower_var = np.array(mean_list_id_mean_list) - np.array(mean_list_id_var_list)

    if sep_measure:
        ax1.plot(dims_list, mean_list_id_mean_list, label="Id PCA", color = "b", linewidth=3, linestyle= (0, (3, 1, 1, 1, 1, 1)))
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "b")
    else: 
        ax1.plot(dims_list, mean_list_id_mean_list, label="Id PCA", color = "b", linewidth=3, linestyle= (0, (3, 1, 1, 1, 1, 1)))
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "b", linewidth = 2)
            
    # Plot for PCA for reconstructed data
    upper_var = np.array(mean_pca_mean_list) + np.array(mean_pca_var_list)
    lower_var = np.array(mean_pca_mean_list) - np.array(mean_pca_var_list)

    if sep_measure:
        ax1.plot(dims_list, mean_pca_mean_list, label="PCA", color = "#007a33", linewidth=3, linestyle = "dashed")
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.4, color = "#007a33")
    else: 
        ax1.plot(dims_list, mean_pca_mean_list, label="PCA", color = "#007a33", linewidth=3, linestyle = "dashed")
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.4, color = "#007a33", linewidth = 2)
            
    if dist_measure != "dtw":  
        
        # Plot for tsne for transformed data
        upper_var = np.array(transf_mean_list_cpca_real_pca_mean_list) + np.array(transf_mean_list_cpca_real_pca_var_list)
        lower_var = np.array(transf_mean_list_cpca_real_pca_mean_list) - np.array(transf_mean_list_cpca_real_pca_var_list)

        if sep_measure:
            ax2.plot(dims_list, transf_mean_list_cpca_real_pca_mean_list, label="CPCA-real: scaled pca", linestyle="dashdot", color="r", linewidth=3)
            if variance:
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color="r",)
        else: 
            ax2.plot(dims_list, transf_mean_list_cpca_real_pca_mean_list, label="CPCA-real: scaled pca", linestyle="dashdot", color="r", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color="r")
        
        # Plot for umap for transformed data
        upper_var = np.array(transf_mean_list_cpca_real_low_mean_list) + np.array(transf_mean_list_cpca_real_low_var_list)
        lower_var = np.array(transf_mean_list_cpca_real_low_mean_list) - np.array(transf_mean_list_cpca_real_low_var_list)

        if sep_measure:
            ax2.plot(dims_list, transf_mean_list_cpca_real_low_mean_list, label="CPCA-real: scaled low imag", linestyle = "dotted", color="m", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color="m")
        else: 
            ax2.plot(dims_list, transf_mean_list_cpca_real_low_mean_list, label="CPCA-real: scaled low imag", linestyle = "dotted", color="m", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color="m")
            
        # Plot for random projection for transformed data
        upper_var = np.array(transf_mean_list_cpca_imag_mean_list) + np.array(transf_mean_list_cpca_imag_var_list)
        lower_var = np.array(transf_mean_list_cpca_imag_mean_list) - np.array(transf_mean_list_cpca_imag_var_list)

        if sep_measure:
            ax2.plot(dims_list, transf_mean_list_cpca_imag_mean_list, label="CPCA-imag", color = "orange", linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "orange")
        else: 
            ax2.plot(dims_list, transf_mean_list_cpca_imag_mean_list, label="CPCA-imag", color = "orange", linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "orange", linewidth = 2)
        
        # Plot for PCA for transformed data
        upper_var = np.array(transf_mean_list_id_mean_list) + np.array(transf_mean_list_id_var_list)
        lower_var = np.array(transf_mean_list_id_mean_list) - np.array(transf_mean_list_id_var_list)

        if sep_measure:
            ax2.plot(dims_list, transf_mean_list_id_mean_list, label="Id PCA", color = "b", linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "b")
        else: 
            ax2.plot(dims_list, transf_mean_list_id_mean_list, label="Id PCA", color = "b", linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "b", linewidth = 2)

        # Plot for lle for transformed data
        upper_var = np.array(transf_mean_pca_mean_list) + np.array(transf_mean_pca_var_list)
        lower_var = np.array(transf_mean_pca_mean_list) - np.array(transf_mean_pca_var_list)

        if sep_measure:
            ax2.plot(dims_list, transf_mean_pca_mean_list, label="PCA", color="#007a33", linewidth=3, linestyle = "dashed")
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color="#007a33")
        else: 
            ax2.plot(dims_list, transf_mean_pca_mean_list, label="PCA", color="#007a33", linewidth=3, linestyle = "dashed")
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color="#007a33")
        
    if dist_measure != "dtw":
        fontsize = plot_dict["lineplots"]["fontsize"]
        fontdist = plot_dict["lineplots"]["font_dist"]
        fontlegend = plot_dict["lineplots"]["font_legend"]
    else:
        fontsize = plot_dict["lineplots"]["dtw_fontsize"]
        fontdist = plot_dict["lineplots"]["dtw_fontdist"]
        fontlegend = plot_dict["lineplots"]["dtw_font_legend"]
    
    ax1.set_xlabel("Number of neighbours", fontsize = fontsize - fontdist)
    ax1.set_ylabel("Avg. dev. from \n original data", fontsize = fontsize - fontdist)
    ax1.set_title("Results for Reconstruction of 10 runs", fontsize = fontsize)
    ax1.legend(fontsize = fontlegend, bbox_to_anchor=(1.06, 1))
    ax1.tick_params(direction='out', labelsize=fontsize - fontdist)
    ax1.set_axisbelow(True)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.yaxis.get_offset_text().set_fontsize(fontsize - fontdist)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.grid(axis='y', linestyle="dashed", color="silver", linewidth=1.5)
    
    ax2.set_xlabel("Number of neighbours", fontsize = fontsize - fontdist)
    ax2.set_ylabel("Avg. dev. from \n original data", fontsize = fontsize - fontdist)
    ax2.set_title("Results for Transformations of 10 runs", fontsize = fontsize)
    ax2.legend(fontsize = fontlegend, bbox_to_anchor=(1.06, 1))
    ax2.tick_params(direction='out', labelsize=fontsize - fontdist)
    ax2.set_axisbelow(True)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.yaxis.get_offset_text().set_fontsize(fontsize - fontdist)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.grid(axis='y', linestyle="dashed", color="silver", linewidth=1.5)
        
    # plt.suptitle("Order " + order + " measure " + str(k) + " neighbours for " + str(num_lines) + " lines a " + str(num_points) + " and jitter " + str(jitter_bound))
    fig.tight_layout(h_pad = 4)
    # plt.box(False)
    plt.savefig("./plots/generated/multiple_runs/cpca/" + str(order) + "/num_neigh/" + str(num_lines) + "lines_" + str(num_points) + "points_" + str(cut_k) + "neighbours_" + str(dist_measure) + ".pdf", bbox_inches="tight")

    plt.show()

def cpca_compute_multiple_num_neigh_vs_dyn_low_dims(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, jitter_bound, num_points, num_lines, k, num_runs):
    
    assert dist_measure not in ["dtw", "scalar_product"], "Error: " + str(dist_measure) + " is not suitable"
    
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    
    low_dim_list = range(2, 11)
    
    seed_list = list(range(num_runs))

    already_low_pickled = find_in_pickle_all_without_seed("./pickled_results/cpca_num_neigh_vs_dyn_low_dims.pickle", [dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, low_dim_list])
    
    slurm_already_low_pickled = find_in_pickle_all_without_seed("./pickled_results/slurm/cpca_num_neigh_vs_dyn_low_dims.pickle", [dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, low_dim_list])

    if already_low_pickled is None:
        already_low_pickled = slurm_already_low_pickled
    
    try:
        already_pickled_low_seeds = [r["parameters"][0] for r in already_low_pickled]
    except:
        already_pickled_low_seeds = []

    for s in seed_list:
        sep_points, separated = generate_correct_ordered_data(order, num_points, num_lines, dimensions, jitter_bound, bounds, parallel_start_end)
        
        # Compute neighbours
        vectors = get_neighbor_vector(sep_points, separated)
        
        if s not in already_pickled_low_seeds:
            logging.info("Computing for seed " + str(s))
            cpca_compute_dyn_low_dim_vs_num_neigh(s, sep_points, separated, vectors, dimensions, k, order, num_lines, num_points, bounds, dist_measure, jitter_bound, parallel_start_end, sep_measure)
        else: 
            logging.info("Already computed results for seed " + str(s))
            

def cpca_compute_dyn_low_dim_vs_num_neigh(seed, sep_points, separated, vectors, start_dim, k, order, num_lines, num_points_per_line, bounds, dist_measure, jitter, parallel_start_end, sep_measure):
    dim_list = range(2, 11)
    already_computed_for_order = find_in_pickle_for_specific_data("./pickled_results/cpca_num_neigh_vs_dyn_low_dims.pickle", order)
    if already_computed_for_order is not None:
        already_computed_parameters = [entry["parameters"] for entry in already_computed_for_order]
    else:
        already_computed_parameters = []
    parameters = [seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k, sep_measure, jitter, parallel_start_end, dim_list]
    if (already_computed_parameters is None) or (parameters not in already_computed_parameters):
        if separated:
            data = flatten(sep_points)
        else:
            data = sep_points
            
        list_pca_means_up_to_k = []
        list_cpca_rea_pca_up_to_k = []
        list_cpca_real_low_up_to_k = []
        list_cpca_imag_means_up_to_k = []
        list_id_means_up_to_k = []
        
        list_transf_pca_means_up_to_k = []
        list_transf_cpca_rea_pca_up_to_k = []
        list_transf_cpca_real_low_up_to_k = []
        list_transf_cpca_imag_means_up_to_k = []
        list_transf_id_means_up_to_k = []
        
        logging.info("Dimensions to be computed " + str(dim_list))
        
        complex_data = add_complex_vector(sep_points, separated)
        if separated:
            id_data = flatten(add_id(sep_points, separated))
        else:
            id_data = add_id(sep_points, separated)
        
        for target_dim in dim_list:
            logging.info("Compute dimension " + str(target_dim))
            logging.info("Compute PCA")
            # Transform and reconstruct data with PCA
            reconstr_data, transf_data = pca_reconstruction(data, target_dim, "complex_pca", None, None)

            # Transform and reconstruct data with cpca and scale to original pca
            logging.info("Get vector data")
            reconstr_cpca_real_pca_data, transf_cpca_real_pca_data = complex_PCA.pca_reconstruction(complex_data, target_dim, "complex_pca_svd_complex", "like_original_pca", None)
            
            # Transform and reconstruct data with cpca and scale to lowest imaginary part
            logging.info("Get vector data")
            reconstr_cpca_real_low_data, transf_cpca_real_low_data = complex_PCA.pca_reconstruction(complex_data, target_dim, "complex_pca_svd_complex", "low_imag", None)
            
            # Transform and reconstruct data with cpca and scale to lowest imaginary part
            logging.info("Get vector data")
            reconstr_cpca_imag_data, transf_cpca_imag_data = complex_PCA.pca_reconstruction(complex_data, target_dim, "complex_pca_svd_complex_imag_pc", "low_imag", None)
            
            # Add Ids and reconstruct and transform data
            reconstr_id_data, transf_id_data = pca_reconstruction(id_data, target_dim + 1, "complex_pca", None, None)
            reconstr_id_data = [elem[:-1] for elem in reconstr_id_data]
            transf_id_data = [elem[:-1] for elem in transf_id_data]
            logging.info("Measure other methods")
            
            if not sep_measure:
                reconstr_exp_values, reconstr_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_data, dist_measure, k)
                    
                reconstr_exp_cpca_real_pca_values, reconstr_cpca_real_pca_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_cpca_real_pca_data, dist_measure, k)
                reconstr_exp_cpca_low_values, reconstr_cpca_low_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_cpca_real_low_data, dist_measure, k)
                reconstr_exp_cpca_imag_values, reconstr_cpca_imag_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_cpca_imag_data, dist_measure, k)
                reconstr_exp_id_values, reconstr_id_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_id_data, dist_measure, k)
            
                transf_exp_values, transf_vars, _ = get_mu_var_by_neighbor_num(data, transf_data, dist_measure, k)
                transf_exp_cpca_real_pca_values, transf_cpca_real_pca_vars, _ = get_mu_var_by_neighbor_num(data, transf_cpca_real_pca_data, dist_measure, k)
                transf_exp_cpca_low_values, transf_cpca_low_vars, _ = get_mu_var_by_neighbor_num(data, transf_cpca_real_low_data, dist_measure, k)
                transf_exp_cpca_imag_values, transf_cpca_imag_vars, _ = get_mu_var_by_neighbor_num(data, transf_cpca_imag_data, dist_measure, k)
                transf_exp_id_values, transf_id_vars, _ = get_mu_var_by_neighbor_num(data, transf_id_data, dist_measure, k)

            elif sep_measure:
                sep_rec_data = [reconstr_data[i:i + num_points_per_line] for i in range(0, len(reconstr_data), num_points_per_line)]
                sep_rec_cpca_real_pca_data = [reconstr_cpca_real_pca_data[i:i + num_points_per_line] for i in range(0, len(reconstr_cpca_real_pca_data), num_points_per_line)]
                sep_rec_cpca_real_low_data = [reconstr_cpca_real_low_data[i:i + num_points_per_line] for i in range(0, len(reconstr_cpca_real_low_data), num_points_per_line)]
                sep_rec_cpca_imag_data = [reconstr_cpca_imag_data[i:i + num_points_per_line] for i in range(0, len(reconstr_cpca_imag_data), num_points_per_line)]
                sep_rec_cpca_id_data = [reconstr_id_data[i:i + num_points_per_line] for i in range(0, len(reconstr_id_data), num_points_per_line)]
                
                reconstr_exp_values, reconstr_vars, _ = measure_sep_structures(sep_points, sep_rec_data, dist_measure, k)
                reconstr_exp_cpca_real_pca_values, reconstr_cpca_real_pca_vars, _ = measure_sep_structures(sep_points, sep_rec_cpca_real_pca_data, dist_measure, k)
                reconstr_exp_cpca_low_values, reconstr_cpca_low_vars, _ = measure_sep_structures(sep_points, sep_rec_cpca_real_low_data, dist_measure, k)
                reconstr_exp_cpca_imag_values, reconstr_cpca_imag_vars, _ = measure_sep_structures(sep_points, sep_rec_cpca_imag_data, dist_measure, k)
                print("pca values ", reconstr_exp_values)
                reconstr_exp_id_values, reconstr_id_vars, _ = measure_sep_structures(sep_points, sep_rec_cpca_id_data, dist_measure, k)
                print("Id values ", reconstr_exp_id_values)

                sep_transf_data = [transf_data[i:i + num_points_per_line] for i in range(0, len(transf_data), num_points_per_line)]
                sep_transf_cpca_real_pca_data = [transf_cpca_real_pca_data[i:i + num_points_per_line] for i in range(0, len(transf_cpca_real_pca_data), num_points_per_line)]
                sep_transf_cpca_real_low_data = [transf_cpca_real_low_data[i:i + num_points_per_line] for i in range(0, len(transf_cpca_real_low_data), num_points_per_line)]
                sep_transf_cpca_imag_data = [transf_cpca_imag_data[i:i + num_points_per_line] for i in range(0, len(transf_cpca_imag_data), num_points_per_line)]
                sep_transf_cpca_id_data = [transf_id_data[i:i + num_points_per_line] for i in range(0, len(transf_id_data), num_points_per_line)]

                transf_exp_values, transf_vars, _ = measure_sep_structures(sep_points, sep_transf_data, dist_measure, k)
                transf_exp_cpca_real_pca_values, transf_cpca_real_pca_vars, _ = measure_sep_structures(sep_points, sep_transf_cpca_real_pca_data, dist_measure, k)
                transf_exp_cpca_low_values, transf_cpca_low_vars, _ = measure_sep_structures(sep_points, sep_transf_cpca_real_low_data, dist_measure, k)
                transf_exp_cpca_imag_values, transf_cpca_imag_vars, _ = measure_sep_structures(sep_points, sep_transf_cpca_imag_data, dist_measure, k)
                transf_exp_id_values, transf_id_vars, _ = measure_sep_structures(sep_points, sep_transf_cpca_id_data, dist_measure, k)

            # Append all
            list_pca_means_up_to_k.append(reconstr_exp_values)
            list_cpca_rea_pca_up_to_k.append(reconstr_exp_cpca_real_pca_values)
            list_cpca_real_low_up_to_k.append(reconstr_exp_cpca_low_values)
            list_cpca_imag_means_up_to_k.append(reconstr_exp_cpca_imag_values)
            list_id_means_up_to_k.append(reconstr_exp_id_values)
            
            list_transf_pca_means_up_to_k.append(transf_exp_values)
            list_transf_cpca_rea_pca_up_to_k.append(transf_exp_cpca_real_pca_values)
            list_transf_cpca_real_low_up_to_k.append(transf_exp_cpca_low_values)
            list_transf_cpca_imag_means_up_to_k.append(transf_exp_cpca_imag_values)
            list_transf_id_means_up_to_k.append(transf_exp_id_values)
            
        logging.info("Finished!")
        pickle_cpca_results_avg_dev_vs_dyn_dim("./pickled_results/cpca_num_neigh_vs_dyn_low_dims.pickle", [seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k, sep_measure, jitter, parallel_start_end, dim_list], [list_pca_means_up_to_k, list_cpca_rea_pca_up_to_k, list_cpca_real_low_up_to_k, list_cpca_imag_means_up_to_k, list_id_means_up_to_k, list_transf_pca_means_up_to_k, list_transf_cpca_rea_pca_up_to_k, list_transf_cpca_real_low_up_to_k, list_transf_cpca_imag_means_up_to_k, list_transf_id_means_up_to_k])
        
    elif parameters in already_computed_parameters:
        logging.info("Parameters already computed for seed " + str(seed))


def cpca_plot_multiple_num_neigh_vs_dyn_low_dims(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, jitter_bound, num_points, num_lines, k, num_runs):
    assert dist_measure != "dtw", "Error: DTW is not suitable"
    assert dist_measure != "scalar_product", "Error: Please use for this function the measure multiple_scalar_product"

    seed_list = list(range(num_runs))
    dim_list = range(2, 11)
    
    parameters = [dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, dim_list]
    result_list = find_in_pickle_all_without_seed("./pickled_results/cpca_num_neigh_vs_dyn_low_dims.pickle", parameters)
    slurm_result_list = find_in_pickle_all_without_seed("./pickled_results/slurm/cpca_num_neigh_vs_dyn_low_dims.pickle", parameters)
    try:
        if len(slurm_result_list) != 0:
            result_list = slurm_result_list
    except:
        pass 
    assert result_list is not None, "Error: There are no pickled results. Please run the multiple computation function"
    
    result_list = [entry for entry in result_list if entry["parameters"][0] in seed_list]
    assert len(result_list) == num_runs
    
    pca_results = [entry["results"][0] for entry in result_list]
    cpca_real_pca_results = [entry["results"][1] for entry in result_list]
    cpca_real_low_results = [entry["results"][2] for entry in result_list]
    
    cpca_imag_results = [entry["results"][3] for entry in result_list]
    ids_results = [entry["results"][4] for entry in result_list]
    transf_pca_results = [entry["results"][5] for entry in result_list]
    
    transf_cpca_real_pca_results = [entry["results"][6] for entry in result_list]
    transf_cpca_real_imag_results = [entry["results"][7] for entry in result_list]
    transf_cpca_imag_results = [entry["results"][8] for entry in result_list]
    transf_ids_results = [entry["results"][9] for entry in result_list]
    
    sorted_pca_results = [[] for d in dim_list]
    sorted_cpca_real_pca_results = [[] for d in dim_list]
    sorted_cpca_real_low_results = [[] for d in dim_list]
    sorted_cpca_imag_results = [[] for d in dim_list]
    sorted_ids_results = [[] for d in dim_list]
    sorted_transf_pca_results = [[] for d in dim_list]
    sorted_transf_cpca_real_pca_results = [[] for d in dim_list]
    sorted_transf_cpca_real_low_results = [[] for d in dim_list]
    sorted_transf_cpca_imag_results = [[] for d in dim_list]
    sorted_transf_ids_results = [[] for d in dim_list]
    
    for s in seed_list:
        for d in range(len(dim_list)):
            sorted_pca_results[d].append(pca_results[s][d])
            sorted_cpca_real_pca_results[d].append(cpca_real_pca_results[s][d])
            sorted_cpca_real_low_results[d].append(cpca_real_low_results[s][d])
            sorted_cpca_imag_results[d].append(cpca_imag_results[s][d])
            sorted_ids_results[d].append(ids_results[s][d])
            sorted_transf_pca_results[d].append(transf_pca_results[s][d])
            sorted_transf_cpca_real_pca_results[d].append(transf_cpca_real_pca_results[s][d])
            sorted_transf_cpca_real_low_results[d].append(transf_cpca_real_imag_results[s][d])
            sorted_transf_cpca_imag_results[d].append(transf_cpca_imag_results[s][d])
            sorted_transf_ids_results[d].append(transf_ids_results[s][d])
    
    mean_pca_results = [np.mean(sorted_pca_results[d], axis = 0) for d in range(len(dim_list))]
    mean_cpca_real_pca_results = [np.mean(sorted_cpca_real_pca_results[d], axis = 0) for d in range(len(dim_list))]
    mean_cpca_real_low_results = [np.mean(sorted_cpca_real_low_results[d], axis = 0) for d in range(len(dim_list))]
    mean_cpca_imag_results = [np.mean(sorted_cpca_imag_results[d], axis = 0) for d in range(len(dim_list))]
    mean_ids_results = [np.mean(sorted_ids_results[d], axis = 0) for d in range(len(dim_list))]
    mean_transf_pca_results = [np.mean(sorted_transf_pca_results[d], axis = 0) for d in range(len(dim_list))]
    mean_transf_cpca_real_pca_results = [np.mean(sorted_transf_cpca_real_pca_results[d], axis = 0) for d in range(len(dim_list))]
    mean_transf_cpca_real_low_results = [np.mean(sorted_transf_cpca_real_low_results[d], axis = 0) for d in range(len(dim_list))]
    mean_transf_cpca_imag_results = [np.mean(sorted_transf_cpca_imag_results[d], axis = 0) for d in range(len(dim_list))]
    mean_transf_ids_results = [np.mean(sorted_transf_ids_results[d], axis = 0) for d in range(len(dim_list))]
    
    all_means = {
        "Reconstr. with PCA" : mean_pca_results, 
        "Transf. with PCA" : mean_transf_pca_results, 
        "Reconstr. with CPCA scaled PCA" : mean_cpca_real_pca_results, 
        "Transf. with CPCA scaled PCA" : mean_transf_cpca_real_pca_results, 
        "Reconstr. with CPCA low imag." : mean_cpca_real_low_results, 
        "Transf. with CPCA low imag." : mean_transf_cpca_real_low_results, 
        "Reconstr. with CPCA with imag." : mean_cpca_imag_results, 
        "Transf. with CPCA with imag." : mean_transf_cpca_imag_results,
        "Reconstr. with Ids" : mean_ids_results, 
        "Transf. with Ids": mean_transf_ids_results
        }
    
    fontsize = plot_dict["heatmaps"]["fontsize"]
    fontdist = plot_dict["heatmaps"]["font_dist"]
    x_tick_labels = list(range(1, k + 1))
    y_tick_labels = dim_list
    legend_min = np.min(flatten([flatten(mean_pca_results), flatten(mean_cpca_real_pca_results), flatten(mean_cpca_real_low_results), flatten(mean_cpca_imag_results), flatten(mean_ids_results)]))
    legend_max = np.max(flatten([flatten(mean_pca_results), flatten(mean_cpca_real_pca_results), flatten(mean_cpca_real_low_results), flatten(mean_cpca_imag_results), flatten(mean_ids_results)]))
    
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    
    fig, ax = plt.subplots(5, 2, sharex= True, sharey = True, figsize=(12, 15))
    fontsize = plot_dict["heatmaps"]["fontsize"]
    fontdist = plot_dict["heatmaps"]["font_dist"]
    for i, key in enumerate(list(all_means.keys())):
        if key == "Reconstruction with Random Projection":
            use_key = "Reconstr. with Rand. Proj."  
        else:
            use_key = key
        sns.heatmap(all_means[key], ax = ax.flat[i], vmin = legend_min, vmax = legend_max, cmap = "mako", square = False, xticklabels = x_tick_labels, yticklabels = y_tick_labels, cbar_kws={"format": formatter})   
        ax.flat[i].set(xticklabels = x_tick_labels, yticklabels = y_tick_labels)
        ax.flat[i].set_title(use_key, fontsize= fontsize)
        ax.flat[i].tick_params(direction='out', labelsize=fontsize - fontdist, rotation = 0)
        if i % 2 == 0:
            ax.flat[i].set_ylabel('Num. of \n Transf. Dim.', fontsize=fontsize - fontdist)
        if i > 7:
            ax.flat[i].set_xlabel('Number of Neighbours', fontsize=fontsize - fontdist)
        cax = ax.flat[i].figure.axes[-1]
        cax.tick_params(labelsize=fontsize - fontdist)
    plt.tight_layout()
    plt.savefig("./plots/generated/multiple_runs/cpca/" + str(order) + "/num_neigh_vs_dyn_low/"+ str(dist_measure) + "_" + str(num_runs)+ "runs_"  + str(num_lines) + "lines_" + str(num_points) + "points_" + str(k) + "neighbours.pdf", bbox_inches="tight")
    plt.show()


def main():
    picked_options = get_picked_options()
    dimensions = picked_options["starting_dimension"]
    target_dimensions = picked_options["target_dimension"]
    order = picked_options["order"]
    bounds = picked_options["bounds"]
    num_lines = picked_options["num_lines"]
    num_points = picked_options["num_points_per_line"]
    dist_measure = picked_options["dist_measure"]
    k = picked_options["compare_k_neighbors"]
    sep_measure = picked_options["sep_measure"]
    jitter_bound = picked_options["jitter_bound"]
    parallel_start_end = picked_options["parallel_start_end"]
    seed = picked_options["seed"]
    jitter_bound = picked_options["jitter_bound"]
    parallel_start_end = picked_options["parallel_start_end"]
    sep_points = picked_options["sep_points"]

    pca_choice = picked_options["pca_choice"]
    scaling_choice = picked_options["scaling_choice"]

    random.seed(seed)
    np.random.seed(seed)
    
    num_runs = 10
    max_k = 49
    variance = False
    if dist_measure == "euclidean":
        variance = True
    
    cpca_multiple_avg_dev_vs_dyn_low_dim(dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, num_runs, parallel_start_end = False)
    # plot_cpca_mean_of_multiple_runs_avg_dev_vs_dyn_low_dim(dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, num_runs, variance, cut_k = None)
    # plot_cpca_mean_of_multiple_runs_avg_dev_vs_dyn_low_dim(dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, num_runs, variance, cut_k = 10)
    
    # cpca_multiple_runs_neighb_error(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, target_dimensions, jitter_bound, num_points, num_lines, max_k, num_runs, scaling_choice = "vector_pca")
    
    # cpca_plot_multiple_runs_neighb_error(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, target_dimensions, jitter_bound, num_points, num_lines, max_k, num_runs, variance, cut_k = None, scaling_choice= "vector_pca")
    
    # cpca_compute_multiple_num_neigh_vs_dyn_low_dims(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, jitter_bound, num_points, num_lines, 10, num_runs)
    # cpca_plot_multiple_num_neigh_vs_dyn_low_dims(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, jitter_bound, num_points, num_lines, 10, num_runs)
    
main()