from comparison_other_dim_reductions import *
from plot_func.dynamic_dimensions import *
from plot_func.plotting import *
from plot_func.heatmaps import *
from data_generation import *
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

def main():
    logging.info("Starting process")

    order_options = ["one_line", "zigzag", "random", "sep_lines", "spiral", "weather", "swiss_roll", "staircase", "helices", "clusters", "connected_zigzag", "parallel_lines", "connected_staircase", "gas", "air_pollution", "spatial"]

    dist_measures_options = ["euclidean", "dtw", "scalar_product"]

    pca_options = {"complex_pca": complex_PCA.ComplexPCA, "complex_pca_svd_real": complex_PCA.ComplexPCA_svd_real, "complex_pca_svd_complex": complex_PCA.ComplexPCA_svd_complex, "complex_pca_svd_complex_imag_pc": complex_PCA.ComplexPCA_svd_complex_imag_pc}

    scaling_vectors_option = ["like_original_pca", "like_vector_pca", "low_imag"]

    when_shuffle_options = ["before_pca", "before_transform", None]
    standardise = False


    reconstruction_error_real = True

    # picked_options = get_picked_options()
    # dimensions = picked_options["starting_dimension"]
    # target_dimensions = picked_options["target_dimension"]
    # order = picked_options["order"]
    # bounds = picked_options["bounds"]
    # num_lines = picked_options["num_lines"]
    # num_points = picked_options["num_points_per_line"]
    # dist_measure = picked_options["dist_measure"]
    # k = picked_options["compare_k_neighbors"]
    # sep_measure = picked_options["sep_measure"]
    # jitter_bound = picked_options["jitter_bound"]
    # seed = picked_options["seed"]
    # jitter_bound = picked_options["jitter_bound"]
    # parallel_start_end = picked_options["parallel_start_end"]
    # sep_points = picked_options["sep_points"]

    # pca_choice = picked_options["pca_choice"]
    # scaling_choice = picked_options["scaling_choice"]
    
    dimensions = 1832
    target_dimensions = 2
    order = "air_pollution"
    bounds = [-918.0, 4379.0]
    num_lines = 1
    num_points = 4380
    dist_measure = "euclidean"
    k = 1
    sep_measure = False
    jitter_bound = None
    parallel_start_end = False
    seed = 0
    jitter_bound = 0
    
    random.seed(seed)
    # np.random.seed(seed)

    # compute_dyn_low_dims_vs_avg_dev(seed, dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, num_runs_rand= 10)
    num_runs = 1
    max_k = 5
    num_rand_runs = 10
    # multiple_runs_neighb_error(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, target_dimensions, jitter_bound, num_points, num_lines, max_k, num_runs)

    ## PLOTS
    variance = False
    if dist_measure == "euclidean":
        variance = True

    # Dynamic dimensions plots with vector PCA
    # plot_reconstr_dyn_high_dims_vs_avg_dev(seed, target_dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end)
    # plot_transf_dyn_high_dims_vs_avg_dev(seed, target_dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end)

    ## These plots are used for real world analysis
    # plot_reconstr_dyn_low_dims_vs_avg_dev(seed, dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, variance = False)
    # plot_transf_dyn_low_dims_vs_avg_dev(seed, dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, variance = False)
    
    # plot_rec_neighborh_error(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, target_dimensions, jitter_bound, num_points, num_lines, max_k, variance, cut_k = None, scaling_choice = "vector_pca", sep_points = None)
    
    # Plot multiple runs for dynamic dimensions
    if order in ["zigzag", "one_line", "sep_lines"]:
        num_runs = 10
    else:
        num_runs = 1
    
    cut_k = None
    # multiple_runs(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, target_dimensions, jitter_bound, num_points, num_lines, k)
    # multiple_runs_transf_all_dim_red_techniques(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, target_dimensions, jitter_bound, num_points, num_lines, k, num_runs)

    # plot_mean_of_multiple_runs_avg_dev_vs_dyn_high_dim(target_dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, num_runs, variance)
    # plot_mean_of_multiple_runs_avg_dev_vs_dyn_low_dim(dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, num_runs, variance, cut_k)
    cut_k = 10
    # plot_mean_of_multiple_runs_avg_dev_vs_dyn_low_dim(dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, num_runs, variance, cut_k)

    k = 5
    # multiple_runs_neighb_error(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, target_dimensions, jitter_bound, num_points, num_lines, k, num_runs)
    
    cut_k = 5
    # plot_multiple_runs_neighb_error(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, target_dimensions, jitter_bound, num_points, num_lines, k, num_runs, variance, cut_k)
    
    k = 5
    # compute_multiple_num_neigh_vs_dyn_low_dims(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, jitter_bound, num_points, num_lines, k, num_runs, num_rand_runs)
    if dist_measure == "scalar_product":
        dist_measure = "multiple_scalar_product"
    plot_multiple_num_neigh_vs_dyn_low_dims(order, parallel_start_end, sep_measure, "euclidean", bounds, dimensions, jitter_bound, num_points, num_lines, k, num_runs, only_pcas=True)
    plot_multiple_num_neigh_vs_dyn_low_dims(order, parallel_start_end, sep_measure, "euclidean", bounds, dimensions, jitter_bound, num_points, num_lines, k, num_runs, only_pcas=False)
    
    plot_multiple_num_neigh_vs_dyn_low_dims(order, parallel_start_end, sep_measure, "multiple_scalar_product", bounds, dimensions, jitter_bound, num_points, num_lines, k, num_runs, only_pcas=True)
    plot_multiple_num_neigh_vs_dyn_low_dims(order, parallel_start_end, sep_measure, "multiple_scalar_product", bounds, dimensions, jitter_bound, num_points, num_lines, k, num_runs, only_pcas=False)
    
    num_lines = 10
    # compute_multiple_num_lines_vs_num_neigh(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, jitter_bound, num_points, num_lines, k, num_runs, target_dimensions)
    
    # plot_num_lines_vs_num_neigh(dimensions, k, order, num_lines, num_points, bounds, dist_measure, jitter_bound, parallel_start_end, sep_measure, target_dimensions, num_runs)
    # # Scalar Product
    seed = 108
    # box_plot_scalar_product(order, dimensions, target_dimensions, num_lines, num_points, bounds, jitter_bound, parallel_start_end, seed = seed)

    # box_plot_multiple_scalar_product(order, dimensions, target_dimensions, num_lines, num_points, bounds, jitter_bound, parallel_start_end, num_runs)


main()
