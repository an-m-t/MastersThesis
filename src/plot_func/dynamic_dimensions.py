from structure_measures import *
import matplotlib.pyplot as plt
from data_generation import *
from general_func import *
from complex_PCA import *
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["axes.formatter.useoffset"] = False
matplotlib.rcParams['axes.titlepad'] = 20

plot_dict = {
    "lineplots" : {
        "fontsize" : 36,
        "font_dist": 6,
        "font_legend": 23,
        "dtw_fontsize": 70,
        "dtw_fontdist": 5,
        "dtw_font_legend": 40
    },
    "heatmaps" : {
        "fontsize" : 20,
        "font_dist" : 0.5
    }
}


def plot_reconstr_dyn_high_dims_vs_avg_dev(seed, target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False):
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

    pca_mean_var, vector_mean_var, rand_proj_means_vars, _, _, _ = compute_dyn_high_dims_vs_avg_dev(seed, target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False)

    dim_list = generate_integer_list_dyn_high(target_dim)

    # Plot for random projection for reconstructed data
    upper_var = np.array(rand_proj_means_vars[0]) + np.array(rand_proj_means_vars[1])
    lower_var = np.array(rand_proj_means_vars[0]) - np.array(rand_proj_means_vars[1])
    
    if sep_measure:
        plt.plot(dim_list, rand_proj_means_vars[0], label="Expected Value: Random Projection", color = "m")
        plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.3, color = "m")
    else:
        plt.plot(dim_list, rand_proj_means_vars[0], label="Expected Value: Random Projection", color = "m")
        plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.3, color = "m", linewidth = 2) 

    # Plot for PCA for reconstructed data
    upper_var = np.array(pca_mean_var[0]) + np.array(pca_mean_var[1])
    lower_var = np.array(pca_mean_var[0]) - np.array(pca_mean_var[1])

    if sep_measure:
        plt.plot(dim_list, pca_mean_var[0], label="Expected Value: PCA", color = "c")
        plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.3, color = "c")
    else: 
        plt.plot(dim_list, pca_mean_var[0], label="Expected Value: PCA", color = "c")
        plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.6, color = "c", linewidth = 2)

    # Plot for Vector PCA for reconstructed data
    upper_var = np.array(vector_mean_var[0]) + np.array(vector_mean_var[1])
    lower_var = np.array(vector_mean_var[0]) - np.array(vector_mean_var[1])
    
    if sep_measure:
        plt.plot(dim_list, vector_mean_var[0], label="Expected Value: Vector PCA", color = "orange")
        plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.3, color = "orange")
    else:
        plt.plot(dim_list, vector_mean_var[0], label="Expected Value: Vector PCA", color = "orange")
        plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.6, color = "orange", linewidth = 2)

    plt.xlabel("Number of dimensions reduced to " + str(target_dim) + " and reconstructed")
    plt.ylabel("Average deviation from original data")
    plt.title("Reconstructed: Average devation vs. dynamic dimension for original space")
    plt.legend()
    plt.show()


def plot_transf_dyn_high_dims_vs_avg_dev(seed, target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False):
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
    _, _, _, transf_pca_mean_var, transf_vector_mean_var, transf_random_proj_mean_var = compute_dyn_high_dims_vs_avg_dev(seed, target_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False)

    dims_list = generate_integer_list_dyn_high(target_dim)

    # Plot for PCA for reconstructed data
    upper_var = np.array(transf_pca_mean_var[0]) + np.array(transf_pca_mean_var[1])
    lower_var = np.array(transf_pca_mean_var[0]) - np.array(transf_pca_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_pca_mean_var[0], label="Expected Value: PCA", color = "c")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "c")
    else: 
        plt.plot(dims_list, transf_pca_mean_var[0], label="Expected Value: PCA", color = "c")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.6, color = "c", linewidth = 2)

    # Plot for Vector PCA for transformed data
    upper_var = np.array(transf_vector_mean_var[0]) + np.array(transf_vector_mean_var[1])
    lower_var = np.array(transf_vector_mean_var[0]) - np.array(transf_vector_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_vector_mean_var[0], label="Expected Value: Vector PCA", color = "orange")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "orange")
    else: 
        plt.plot(dims_list, transf_vector_mean_var[0], label="Expected Value: Vector PCA", color = "orange")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.6, color = "orange", linewidth = 2)

    # Plot for random projection for transformed data
    upper_var = np.array(transf_random_proj_mean_var[0]) + np.array(transf_random_proj_mean_var[1])
    lower_var = np.array(transf_random_proj_mean_var[0]) - np.array(transf_random_proj_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_random_proj_mean_var[0], label="Expected Value: Random Projection", color = "m")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "m")
    else: 
        plt.plot(dims_list, transf_random_proj_mean_var[0], label="Expected Value: Random Projection", color = "m")
        plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "m", linewidth = 2)

    plt.xlabel("Number of dimensions reduced to " + str(target_dim))
    plt.ylabel("Average deviation from original data")
    plt.title("Transformed: Average deviation vs. dynamic dimension for original space")
    plt.legend()
    plt.set_yscale('log')
    plt.show()


def plot_reconstr_dyn_low_dims_vs_avg_dev(seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False, variance = True):
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
    dim_list = generate_integer_list_dyn_low(start_dim, num_lines, order)
    fig, ax = plt.subplots(figsize=(14,10))
    
    pca_mean_var, vector_mean_var, rand_proj_means_vars, _, _, _ = compute_dyn_low_dims_vs_avg_dev(seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end)
    
    # Plot for random projection for reconstructed data
    upper_var = np.array(rand_proj_means_vars[0]) + np.array(rand_proj_means_vars[1])
    lower_var = np.array(rand_proj_means_vars[0]) - np.array(rand_proj_means_vars[1])
    
    if sep_measure:
        plt.plot(dim_list, rand_proj_means_vars[0], label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
        if variance:
            plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.3, color = "#ff42c8", linestyle="dashdot", linewidth=3)
    else:
        plt.plot(dim_list, rand_proj_means_vars[0], label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
        if variance:
            plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.3, color = "#ff42c8")

    # Plot for PCA for reconstructed data
    upper_var = np.array(pca_mean_var[0]) + np.array(pca_mean_var[1])
    lower_var = np.array(pca_mean_var[0]) - np.array(pca_mean_var[1])

    if sep_measure:
        plt.plot(dim_list, pca_mean_var[0], label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
        if variance:
            plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.3, color = "#007a33", linestyle="dashed", linewidth=3)
    else: 
        plt.plot(dim_list, pca_mean_var[0], label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
        if variance:
            plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.3, color = "#007a33")

    # Plot for Vector PCA for reconstructed data
    upper_var = np.array(vector_mean_var[0]) + np.array(vector_mean_var[1])
    lower_var = np.array(vector_mean_var[0]) - np.array(vector_mean_var[1])
    
    if sep_measure:
        plt.plot(dim_list, vector_mean_var[0], label="PCA*", color = "gold", linewidth = 3)
        if variance:    
            plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.3, color = "gold")
    else:
        plt.plot(dim_list, vector_mean_var[0], label="PCA*", color = "gold", linewidth = 3)
        if variance:    
            plt.fill_between(dim_list, lower_var, upper_var, alpha = 0.3, color = "gold", linewidth = 3)

    fontsize = plot_dict["lineplots"]["fontsize"]
    fontdist = plot_dict["lineplots"]["font_dist"]
    fontlegend = plot_dict["lineplots"]["font_legend"]
    
    ax.legend(loc=(1.06, 0.6), fontsize = fontlegend, borderaxespad=0)
    ax.set_xlabel("Dimension " + str(start_dim) + " reduced to", fontsize = fontsize - fontdist)
    ax.set_ylabel("Avg. dev. from \n orig. data", fontsize = fontsize - fontdist)
    ax.set_title("Results for Reconstruction", fontsize = fontsize)
    ax.tick_params(direction='out', labelsize=fontsize -fontdist)
    ax.set_axisbelow(True)
    if order =="flights":
        ax.set_xticks([d for d in dim_list if d % 2 == 0])
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(fontsize - fontdist)
    range_max = max(flatten([vector_mean_var[0], pca_mean_var[0], rand_proj_means_vars[0]]))
    if dist_measure not in ["multiple_scalar_product", "scalar_product"]:
        variance_max = max(flatten([vector_mean_var[1], pca_mean_var[1], rand_proj_means_vars[1]]))
        range_max += variance_max
        if order == "flights":
            range_max -= 500
    next_higher = pow(10, len(str(round(range_max))))
    if range_max < next_higher - int(next_higher / 2):
            next_higher = next_higher - int(next_higher / 2)
    parts = 5
    ax.set_yticks(range(0, next_higher, int(next_higher / parts)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='y', linestyle="dashed", color="silver", linewidth=1.5)
    fig.tight_layout()
        # fig.set_figwidth(14)
        # fig.set_figheight(8)
        
    if dist_measure in ["multiple_scalar_product", "scalar_product"] or (dist_measure == "euclidean" and order == "russell2000_stock") or (dist_measure == "euclidean" and order == "flights"):
        ax.set_yscale('asinh')
        ax.set_ylim(ymin=0)
        # ax.set_xticks([d for d in dim_list if d % 50 == 0])
        # ax.set_xticks([d for d in dim_list if d % 50 == 0])
    fig.tight_layout()
    if order in ["CMAPSS", "flights", "air_pollution", "russell2000_stock"]:
        plt.savefig("./plots/real-world/" + str(order) + "/avg_dev_vs_dyn_low/reconstructed_" + str(num_lines) + "lines_" + str(num_points_per_line) + "points_" + str(dist_measure) + ".pdf", bbox_inches="tight")
    else:
        plt.savefig("./plots/generated/" + str(order) + "/avg_dev_vs_dyn_low/reconstructed_" + str(num_lines) + "lines_" + str(num_points_per_line) + "points_" + str(dist_measure) + ".pdf", bbox_inches="tight")
    plt.show()   


def plot_transf_dyn_low_dims_vs_avg_dev(seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end = False, variance = True):
    assert dist_measure != "dtw", "Error: DTW only works for data with the same dimensionality"
    dims_list = generate_integer_list_dyn_low(start_dim, num_lines, order)
    
    _, _, _, transf_pca_mean_var, transf_vector_mean_var, transf_random_proj_mean_var = compute_dyn_low_dims_vs_avg_dev(seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k_neigh, sep_measure, jitter, parallel_start_end)
    
    fig, ax = plt.subplots(figsize=(14,10))
    # Plot for random projection for transformed data
    upper_var = np.array(transf_random_proj_mean_var[0]) + np.array(transf_random_proj_mean_var[1])
    lower_var = np.array(transf_random_proj_mean_var[0]) - np.array(transf_random_proj_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_random_proj_mean_var[0], label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
        if variance:
            plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "#ff42c8")
    else: 
        plt.plot(dims_list, transf_random_proj_mean_var[0], label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
        if variance:
            plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "#ff42c8")

    # Plot for PCA for reconstructed data
    upper_var = np.array(transf_pca_mean_var[0]) + np.array(transf_pca_mean_var[1])
    lower_var = np.array(transf_pca_mean_var[0]) - np.array(transf_pca_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_pca_mean_var[0], label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
        if variance:
            plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "#007a33")
    else: 
        plt.plot(dims_list, transf_pca_mean_var[0], label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
        if variance:
            plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "#007a33")

    # Plot for Vector PCA for transformed data
    upper_var = np.array(transf_vector_mean_var[0]) + np.array(transf_vector_mean_var[1])
    lower_var = np.array(transf_vector_mean_var[0]) - np.array(transf_vector_mean_var[1])

    if sep_measure:
        plt.plot(dims_list, transf_vector_mean_var[0], label="PCA*", color = "gold", linewidth = 3)
        if variance:
            plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "gold")
    else: 
        plt.plot(dims_list, transf_vector_mean_var[0], label="PCA*", color = "gold", linewidth = 3)
        if variance:
            plt.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "gold")
    fontsize = plot_dict["lineplots"]["fontsize"]
    fontdist = plot_dict["lineplots"]["font_dist"]
    fontlegend = plot_dict["lineplots"]["font_legend"]
    ax.legend(loc=(1.06, 0.6), fontsize = fontlegend, borderaxespad=0)
    ax.set_xlabel("Dimension " + str(start_dim) + " reduced to", fontsize = fontsize - fontdist)
    ax.set_ylabel("Avg. dev. from \n orig. data", fontsize = fontsize - fontdist)
    ax.set_title("Results for Transformation", fontsize = fontsize)
    ax.tick_params(direction='out', labelsize=fontsize -fontdist)
    ax.set_axisbelow(True)
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(fontsize - fontdist)
    range_max = max(flatten([transf_vector_mean_var[0], transf_pca_mean_var[0], transf_random_proj_mean_var[0]]))
    if dist_measure not in ["multiple_scalar_product", "scalar_product"]:
        variance_max = max(flatten([transf_vector_mean_var[1], transf_pca_mean_var[1], transf_random_proj_mean_var[1]]))
        range_max += variance_max -500
    next_higher = pow(10, len(str(round(range_max))))
    if range_max < next_higher - int(next_higher / 2):
            next_higher = next_higher - int(next_higher / 2)
    parts = 5
    ax.set_yticks(range(0, next_higher, int(next_higher / parts)))
    if order =="flights":
        ax.set_xticks([d for d in dims_list if d % 2 == 0])
        ax.set_ylim(0)
        ax.set_yticks(range(0, 600 - 50, 50))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='y', linestyle="dashed", color="silver", linewidth=1.5)
    fig.tight_layout()
    if dist_measure in ["multiple_scalar_product", "scalar_product"] or (dist_measure == "euclidean" and order == "russell2000_stock") or (dist_measure == "euclidean" and order == "flights"):
        # plt.yscale('log',base=10) 
        ax.set_yscale('asinh')
        ax.set_ylim(ymin=0)
    if order in ["CMAPSS", "flights", "air_pollution", "russell2000_stock"]:
        plt.savefig("./plots/real-world/" + str(order) + "/avg_dev_vs_dyn_low/transformed_" + str(num_lines) + "lines_" + str(num_points_per_line) + "points_" + str(dist_measure) + ".pdf", bbox_inches="tight")
    else:
        plt.savefig("./plots/generated/" + str(order) + "/avg_dev_vs_dyn_low/transformed" + str(num_lines) + "lines_" + str(num_points_per_line) + "points_" + str(dist_measure) + ".pdf", bbox_inches="tight")
    plt.show()