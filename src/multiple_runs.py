from comparison_other_dim_reductions import *
from plot_func.dynamic_dimensions import *
from plot_func.plotting import *
from plot_func.heatmaps import *
from data_generation import *
import pandas as pd
import json
import orderings
import random
from ica import *
from real_data_cleaning import *
from plot_func.dynamic_dimensions import pickle_results_avg_dev_vs_dyn_dim
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as tkr
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["axes.formatter.useoffset"] = False
matplotlib.rcParams['axes.titlepad'] = 26

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
        "fontsize" : 27,
        "font_dist" : 3
    }
}

def multiple_runs(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, target_dimensions, jitter_bound, num_points, num_lines, k):
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    
    high_dim_list = generate_integer_list_dyn_high(target_dimensions)
    low_dim_list = generate_integer_list_dyn_low(dimensions, num_lines, order)

    already_high_computed = find_in_pickle_all_without_seed("./pickled_results/rec_avg_dev_vs_dyn_high_dims.pickle", [target_dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, high_dim_list])

    already_low_computed = find_in_pickle_all_without_seed("./pickled_results/rec_avg_dev_vs_dyn_low_dims.pickle", [dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, low_dim_list])

    try:
        already_computed_high_seeds = [r["parameters"][0] for r in already_high_computed]
    except:
        already_computed_high_seeds = []
    try:
        already_computed_low_seeds = [r["parameters"][0] for r in already_low_computed]
    except:
        already_computed_low_seeds = []
    already_computed_seeds = already_computed_high_seeds + already_computed_low_seeds
    print("Already computed seeds: " + str(set(already_computed_seeds)))

    full_range = list(range(1, 101))
    filtered_range = [num for num in full_range if num not in already_computed_seeds]

    seed_list = random.sample(filtered_range, 10)
    print("New seed list " + str(seed_list))

    for s in seed_list:
        random.seed(s)
        np.random.seed(s)

        # Compute average deviation with dynamic original dimensions
        pca_mean_var, vector_mean_var, rand_proj_means_vars, transf_pca_mean_var, transf_vector_mean_var, transf_random_proj_mean_var = compute_dyn_high_dims_vs_avg_dev(s, target_dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end)

        # Compute average deviation with dynamic transformation dimension
        low_pca_mean_var, low_vector_mean_var, low_rand_proj_means_vars, low_transf_pca_mean_var, low_transf_vector_mean_var, low_transf_random_proj_mean_var = compute_dyn_low_dims_vs_avg_dev(s, dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end)


def multiple_runs_transf_all_dim_red_techniques(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, target_dimensions, jitter_bound, num_points, num_lines, k, num_runs, num_runs_rand = 1):
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    
    high_dim_list = generate_integer_list_dyn_high(target_dimensions)
    low_dim_list = generate_integer_list_dyn_low(dimensions, num_lines, order)
    
    seed_list = list(range(num_runs))

    logging.info("Get already computed seeds")
    already_high_computed = find_in_pickle_all_without_seed("./pickled_results/avg_dev_vs_dyn_high_dims.pickle", [target_dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, high_dim_list])

    already_low_computed = find_in_pickle_all_without_seed("./pickled_results/avg_dev_vs_dyn_low_dims.pickle", [dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, low_dim_list])

    already_high_computed_other_techniques = find_in_pickle_all_without_seed("./pickled_results/other_methods_avg_dev_vs_dyn_high_dims.pickle", [target_dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, high_dim_list])

    already_low_computed_other_techniques = find_in_pickle_all_without_seed("./pickled_results/other_methods_avg_dev_vs_dyn_low_dims.pickle", [dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, low_dim_list])

    try:
        already_computed_high_seeds = [r["parameters"][0] for r in already_high_computed]
    except:
        already_computed_high_seeds = []
    try:
        already_computed_low_seeds = [r["parameters"][0] for r in already_low_computed]
    except:
        already_computed_low_seeds = []

    try:
        already_computed_high_seeds_other_techn = [r["parameters"][0] for r in already_high_computed_other_techniques]
    except:
        already_computed_high_seeds_other_techn = []
    try:
        already_computed_low_seeds_other_techn = [r["parameters"][0] for r in already_low_computed_other_techniques]
    except:
        already_computed_low_seeds_other_techn = []

    already_computed_seeds = already_computed_high_seeds + already_computed_low_seeds + already_computed_high_seeds_other_techn + already_computed_low_seeds_other_techn

    logging.info("Already computed seeds: " + str(set(already_computed_seeds)))

    full_range = list(range(1, 1001))
    filtered_range = [num for num in full_range if num not in already_computed_seeds]

    # seed_list = random.sample(filtered_range, 10)
    print("New seed list " + str(seed_list))

    for s in seed_list:
        random.seed(s)
        np.random.seed(s)
        
        if s not in already_computed_high_seeds and order not in ["russell200_stock", "air_pollution", "flights"]:
            # Compute average deviation with dynamic original dimensions
            pca_mean_var, vector_mean_var, rand_proj_means_vars, transf_pca_mean_var, transf_vector_mean_var, transf_random_proj_mean_var = compute_dyn_high_dims_vs_avg_dev(s, target_dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end)
        else:
            logging.info("Info: Already computed enough runs for high dims")

        if s not in already_computed_high_seeds_other_techn and dist_measure != "dtw" and order not in ["russell200_stock", "air_pollution", "flights"]:
            lle_mean_var, tsne_mean_var, umap_mean_var = compute_lle_tsne_umap_dyn_high_dims_vs_avg_dev(s, target_dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end)
        else:
            logging.info("Info: Already computed enough runs for other methods and high dims")
        
        if s not in already_computed_low_seeds:
            # Compute average deviation with dynamic transformation dimension
            low_pca_mean_var, low_vector_mean_var, low_rand_proj_means_vars, low_transf_pca_mean_var, low_transf_vector_mean_var, low_transf_random_proj_mean_var = compute_dyn_low_dims_vs_avg_dev(s, dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, num_runs_rand = num_runs_rand)
        else:
            logging.info("Already computed enough runs for dy low dims")

        if s not in already_computed_low_seeds_other_techn and dist_measure != "dtw" and order not in ["russell200_stock", "air_pollution", "flights"]:
            lle_mean_var, tsne_mean_var, umap_mean_var = compute_lle_tsne_dyn_low_dims_vs_avg_dev(s, dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end)
        else:
            logging.info("Already computed enough runs for dyn low dims for other methods")
        

def plot_mean_of_multiple_runs_avg_dev_vs_dyn_high_dim(target_dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, num_runs, variance = True):
    
    dims_list = generate_integer_list_dyn_high(target_dimensions)
    
    seed_list = list(range(num_runs))
    
    parameters = [target_dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, dims_list]
    result_list = find_in_pickle_all_without_seed("./pickled_results/avg_dev_vs_dyn_high_dims.pickle", parameters)
    slurm_result_list = find_in_pickle_all_without_seed("./pickled_results/slurm/avg_dev_vs_dyn_high_dims.pickle", parameters)
    other_results = find_in_pickle_all_without_seed("./pickled_results/other_methods_avg_dev_vs_dyn_high_dims.pickle", parameters)
    slurm_other_results = find_in_pickle_all_without_seed("./pickled_results/slurm/other_methods_avg_dev_vs_dyn_high_dims.pickle", parameters)
    
    if len(slurm_result_list) != 0:
        result_list = slurm_result_list
    
    if len(slurm_other_results) != 0:
        other_results = slurm_other_results
    assert result_list is not None, "Error: There are no pickled results for pca, vector pca and random-projections"
    if dist_measure != "dtw":
        assert other_results is not None, "Error: There are no pickled results for lle, tsne and umap"
    result_list = [entry for entry in result_list if entry["parameters"][0] in seed_list]
    other_results = [entry for entry in other_results if entry["parameters"][0] in seed_list]
    
    seeds = [] 
    new_other_results = []
    
    for res in other_results:
        if res["parameters"][0] not in seeds:
            seeds.append(res["parameters"][0])
            new_other_results.append(res)
    other_results = new_other_results
    seeds = [] 
    new_results = []
    
    for res in result_list:
        if res["parameters"][0] not in seeds:
            seeds.append(res["parameters"][0])
            new_results.append(res)
            
    result_list = new_results
    assert len(result_list) == num_runs
    if dist_measure != "dtw":
        assert len(other_results) == num_runs

    pca_mean_list = []
    pca_var_list = []
    vector_mean_list = []
    vector_var_list = []
    random_proj_mean_list = []
    random_proj_var_list = []

    transf_pca_mean_list = []
    transf_pca_var_list = []
    transf_vector_mean_list = []
    transf_vector_var_list = []
    transf_random_proj_mean_list = []
    transf_random_proj_var_list = []
    
    transf_tsne_mean_list = []
    transf_tsne_var_list = []
    transf_lle_mean_list = []
    transf_lle_var_list = []
    transf_umap_mean_list = []
    transf_umap_var_list = []

    for res in result_list:
        pca_mean_list.append(res["pca_mean_var"][0])
        pca_var_list.append(res["pca_mean_var"][1])
        vector_mean_list.append(res["vector_mean_var"][0])
        vector_var_list.append(res["vector_mean_var"][1])
        random_proj_mean_list.append(res["rand_proj_means_vars"][0])
        random_proj_var_list.append(res["rand_proj_means_vars"][1])

        transf_pca_mean_list.append(res["transf_pca_mean_var"][0])
        transf_pca_var_list.append(res["transf_pca_mean_var"][1])
        transf_vector_mean_list.append(res["transf_vector_mean_var"][0])
        transf_vector_var_list.append(res["transf_vector_mean_var"][1])
        transf_random_proj_mean_list.append(res["transf_random_proj_mean_var"][0])
        transf_random_proj_var_list.append(res["transf_random_proj_mean_var"][1])
    if dist_measure != "dtw":
        for res in other_results:
            transf_lle_mean_list.append(res["transf_lle_mean_var"][0])
            transf_lle_var_list.append(res["transf_lle_mean_var"][1])
            
            transf_tsne_mean_list.append(res["transf_tsne_mean_var"][0])
            transf_tsne_var_list.append(res["transf_tsne_mean_var"][1])
            
            transf_umap_mean_list.append(res["transf_umap_mean_var"][0])
            transf_umap_var_list.append(res["transf_umap_mean_var"][1])

        mean_transf_pca_mean_list = np.mean(transf_pca_mean_list, axis=0)
        mean_transf_pca_var_list = np.mean(transf_pca_var_list, axis=0)
        mean_transf_vector_mean_list = np.mean(transf_vector_mean_list, axis=0)
        mean_transf_vector_var_list = np.mean(transf_vector_var_list, axis=0)
        mean_transf_random_proj_mean_list = np.mean(transf_random_proj_mean_list, axis=0)
        mean_transf_random_proj_var_list = np.mean(transf_random_proj_var_list, axis=0)
        
        mean_lle_mean_list = np.mean(transf_lle_mean_list, axis = 0)
        mean_lle_var_list = np.mean(transf_lle_var_list, axis = 0)
        mean_tsne_mean_list = np.mean(transf_tsne_mean_list, axis = 0)
        mean_tsne_var_list = np.mean(transf_tsne_var_list, axis = 0)
        mean_umap_mean_list = np.mean(transf_umap_mean_list, axis = 0)
        mean_umap_var_list = np.mean(transf_umap_var_list, axis = 0)
    
    mean_pca_mean_list = np.mean(pca_mean_list, axis=0)
    mean_pca_var_list = np.mean(pca_var_list, axis=0)
    mean_vector_mean_list = np.mean(vector_mean_list, axis=0)
    mean_vector_var_list = np.mean(vector_var_list, axis=0)
    mean_random_proj_mean_list = np.mean(random_proj_mean_list, axis=0)
    mean_random_proj_var_list = np.mean(random_proj_var_list, axis=0)
        
    if dist_measure == "dtw":
        fig, ax1 = plt.subplots(figsize=(14, 10))
    else:
        fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 10))
        
    # Plot for PCA for reconstructed data
    upper_var = np.array(mean_pca_mean_list) + np.array(mean_pca_var_list)
    lower_var = np.array(mean_pca_mean_list) - np.array(mean_pca_var_list)

    if sep_measure:
        ax1.plot(dims_list, mean_pca_mean_list, label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.4, color = "#007a33")
    else: 
        ax1.plot(dims_list, mean_pca_mean_list, label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.4, color = "#007a33")
        
    # Plot for random projection for reconstructed data
    upper_var = np.array(mean_random_proj_mean_list) + np.array(mean_random_proj_var_list)
    lower_var = np.array(mean_random_proj_mean_list) - np.array(mean_random_proj_var_list)

    if sep_measure:
        ax1.plot(dims_list, mean_random_proj_mean_list, label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "#ff42c8")
    else: 
        ax1.plot(dims_list, mean_random_proj_mean_list, label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "#ff42c8",)
            
    # Plot for Vector PCA for reconstructed data
    upper_var = np.array(mean_vector_mean_list) + np.array(mean_vector_var_list)
    lower_var = np.array(mean_vector_mean_list) - np.array(mean_vector_var_list)

    if sep_measure:
        ax1.plot(dims_list, mean_vector_mean_list, label="PCA*", color = "gold", linewidth = 3)
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.4, color = "gold")
    else: 
        ax1.plot(dims_list, mean_vector_mean_list, label="PCA*", color = "gold", linewidth = 3)
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.4, color = "gold")
    
    if dist_measure != "dtw":        
        # Plot for lle for transformed data
        upper_var = np.array(mean_lle_mean_list) + np.array(mean_lle_var_list)
        lower_var = np.array(mean_lle_mean_list) - np.array(mean_lle_var_list)

        if sep_measure:
            ax2.plot(dims_list, mean_lle_mean_list, label="LLE", linestyle="dotted", color="#e00031", linewidth=3)
            if variance:
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.2, color="#e00031")
        else: 
            ax2.plot(dims_list, mean_lle_mean_list, label="LLE", linestyle="dotted", color="#e00031", linewidth=3)
            if variance:
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.2, color="#e00031")
        
        # Plot for tsne for transformed data
        upper_var = np.array(mean_tsne_mean_list) + np.array(mean_tsne_var_list)
        lower_var = np.array(mean_tsne_mean_list) - np.array(mean_tsne_var_list)

        if sep_measure:
            ax2.plot(dims_list, mean_tsne_mean_list, label="t-SNE", marker=".", color="#9b63f3", linewidth=3)
            if variance:
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.2, color="#9b63f3")
        else: 
            ax2.plot(dims_list, mean_tsne_mean_list, label="t-SNE", marker=".", color="#9b63f3", linewidth=3)
            if variance:
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.2, color="#9b63f3")
        
        # Plot for umap for transformed data
        upper_var = np.array(mean_umap_mean_list) + np.array(mean_umap_var_list)
        lower_var = np.array(mean_umap_mean_list) - np.array(mean_umap_var_list)

        if sep_measure:
            ax2.plot(dims_list, mean_umap_mean_list, label="UMAP", marker="x", color="#2b9de9", linewidth=3)
            if variance:
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.2, color="#2b9de9")
        else: 
            ax2.plot(dims_list, mean_umap_mean_list, label="UMAP", marker="x", color="#2b9de9", linewidth=3)
            if variance:
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.2, color="#2b9de9")

        # Plot for random projection for transformed data
        upper_var = np.array(mean_transf_random_proj_mean_list) + np.array(mean_transf_random_proj_var_list)
        lower_var = np.array(mean_transf_random_proj_mean_list) - np.array(mean_transf_random_proj_var_list)

        if sep_measure:
            ax2.plot(dims_list, mean_transf_random_proj_mean_list, label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
            if variance:
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "#ff42c8")
        else: 
            ax2.plot(dims_list, mean_transf_random_proj_mean_list, label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
            if variance:
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "#ff42c8", linewidth = 2)
        
        # Plot for PCA for transformed data
        upper_var = np.array(mean_transf_pca_mean_list) + np.array(mean_transf_pca_var_list)
        lower_var = np.array(mean_transf_pca_mean_list) - np.array(mean_transf_pca_var_list)

        if sep_measure:
            ax2.plot(dims_list, mean_transf_pca_mean_list, label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
            if variance:
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.4, color = "#007a33")
        else: 
            ax2.plot(dims_list, mean_transf_pca_mean_list, label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
            if variance:
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.4, color = "#007a33", linewidth = 2)
            
        # Plot for PCA* for transformed data
        upper_var = np.array(mean_transf_vector_mean_list) + np.array(mean_transf_vector_var_list)
        lower_var = np.array(mean_transf_vector_mean_list) - np.array(mean_transf_vector_var_list)

        if sep_measure:
            ax2.plot(dims_list, mean_transf_vector_mean_list, label="PCA*", color = "gold", linewidth = 3)
            if variance:
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.4, color = "gold")
        else: 
            ax2.plot(dims_list, mean_transf_vector_mean_list, label="PCA*", color = "gold", linewidth = 3)
            if variance:
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.4, color = "gold")
            
    if dist_measure != "dtw":
        fontsize = plot_dict["lineplots"]["fontsize"]
        fontdist = plot_dict["lineplots"]["font_dist"]
        fontlegend = plot_dict["lineplots"]["font_legend"]
        ax1.legend(bbox_to_anchor=(1.06, 0.8), borderaxespad=0, fontsize = fontlegend)
    else:
        fontsize = plot_dict["lineplots"]["dtw_fontsize"]
        fontdist = plot_dict["lineplots"]["dtw_fontdist"]
        fontlegend = plot_dict["lineplots"]["dtw_font_legend"]
        ax1.legend(fontsize = fontlegend)
        
    ax1.set_xlabel("Number of dimensions reduced to " + str(target_dimensions), fontsize=fontsize - fontdist)
    ax1.set_ylabel("Avg. dev. from \n orig. data", fontsize=fontsize - fontdist)
    ax1.set_title("Results for Reconstruction of " + str(num_runs) + " runs", fontsize = fontsize)
    ax1.tick_params(direction='out', labelsize=fontsize - fontdist)
    ax1.set_axisbelow(True)
    ax1.set_xticks([dims_list[0]] + [d for d in dims_list if d % 10 == 0])
    range_max = max(flatten([mean_pca_mean_list, mean_vector_mean_list, mean_random_proj_mean_list]))
    next_higher = pow(10, len(str(round(range_max))))
    parts = 5
    if range_max < int(next_higher / 4):
            next_higher = int(next_higher / 4)
    ax1.set_yticks(range(0, next_higher, int(next_higher / parts)))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.yaxis.get_offset_text().set_fontsize(fontsize - fontdist)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.grid(axis='y', linestyle="dashed", color="silver", linewidth=1.5)
    
    if dist_measure != "dtw":
        ax2.set_xlabel("Number of dimensions reduced to " + str(target_dimensions), fontsize=fontsize - fontdist)
        ax2.set_ylabel("Avg. dev. from \n orig. data", fontsize=fontsize - fontdist)
        ax2.set_title("Results for Transformation of " + str(num_runs) + " runs", fontsize = fontsize)
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        range_max = max(flatten([mean_lle_mean_list, mean_tsne_mean_list, mean_umap_mean_list, mean_transf_pca_mean_list, mean_transf_random_proj_mean_list, mean_transf_vector_mean_list]))
        ax2.yaxis.get_offset_text().set_fontsize(fontsize - fontdist)
        ax2.legend(bbox_to_anchor=(1.06, 1), borderaxespad=0, fontsize = fontlegend)
        next_higher = pow(10, len(str(round(range_max))))
        ax2.tick_params(direction='out', labelsize=fontsize - fontdist)
        ax2.set_axisbelow(True)
        ax2.set_xticks([dims_list[0]] + [d for d in dims_list if d % 10 == 0])
        parts = 5
        ax2.set_yticks(range(0, next_higher, int(next_higher / parts)))
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.grid(axis='y', linestyle="dashed", color="silver", linewidth=1.5)
        
    # plt.suptitle("Order " + order + " measure " + str(k) + " neighbours for " + str(num_lines) + " lines a " + str(num_points))
    if dist_measure == "dtw":
        fig.tight_layout()
    else:
        fig.tight_layout(h_pad = 4)
        # fig.set_figwidth(14)
        # fig.set_figheight(8)
    plt.savefig("./plots/generated/multiple_runs/" + str(order) + "/avg_dev_vs_dyn_high/" + str(num_lines) + "lines_" + str(num_points) + "points_" + str(k) + "neighbours_" + str(dist_measure) + ".pdf", bbox_inches="tight")
    # ax1.set_yscale("log")
    # ax2.set_yscale("log")
    plt.show()


def plot_mean_of_multiple_runs_avg_dev_vs_dyn_low_dim(start_dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, num_runs, variance = False, cut_k = None):
    dims_list = generate_integer_list_dyn_low(start_dimensions, num_lines, order)
    
    seed_list = list(range(num_runs))
    
    parameters = [start_dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, dims_list]
    result_list = find_in_pickle_all_without_seed("./pickled_results/avg_dev_vs_dyn_low_dims.pickle", parameters)
    
    slurm_result_list = find_in_pickle_all_without_seed("./pickled_results/slurm/avg_dev_vs_dyn_low_dims.pickle", parameters)
    other_results = find_in_pickle_all_without_seed("./pickled_results/other_methods_avg_dev_vs_dyn_low_dims.pickle", parameters)
    
    slurm_other_results = find_in_pickle_all_without_seed("./pickled_results/slurm/other_methods_avg_dev_vs_dyn_low_dims.pickle", parameters)
    
    assert result_list is not None or slurm_result_list is not None, "Error: There are no pickled results for pca, vector pca and random-projections"
    if dist_measure != "dtw":
        assert other_results is not None or slurm_other_results is not None , "Error: There are no pickled results for lle, tsne and umap"
    
    if len(slurm_result_list) != 0:
        result_list = slurm_result_list
    if len(slurm_other_results) != 0:
        other_results = slurm_other_results
    
    result_list = [entry for entry in result_list if entry["parameters"][0] in seed_list]
    other_results = [entry for entry in other_results if entry["parameters"][0] in seed_list]
    seeds = [] 
    new_other_results = []
    
    for res in other_results:
        if res["parameters"][0] not in seeds:
            seeds.append(res["parameters"][0])
            new_other_results.append(res)
            
    other_results = new_other_results  
    assert len(result_list) == num_runs
    if dist_measure != "dtw":
        assert len(other_results) == num_runs

    pca_mean_list = []
    pca_var_list = []
    vector_mean_list = []
    vector_var_list = []
    random_proj_mean_list = []
    random_proj_var_list = []

    transf_pca_mean_list = []
    transf_pca_var_list = []
    transf_vector_mean_list = []
    transf_vector_var_list = []
    transf_random_proj_mean_list = []
    transf_random_proj_var_list = []
    
    transf_tsne_mean_list = []
    transf_tsne_var_list = []
    transf_lle_mean_list = []
    transf_lle_var_list = []
    transf_umap_mean_list = []
    transf_umap_var_list = []

    for res in result_list:
        pca_mean_list.append(res["pca_mean_var"][0])
        pca_var_list.append(res["pca_mean_var"][1])
        vector_mean_list.append(res["vector_mean_var"][0])
        vector_var_list.append(res["vector_mean_var"][1])
        random_proj_mean_list.append(res["rand_proj_means_vars"][0])
        random_proj_var_list.append(res["rand_proj_means_vars"][1])

        transf_pca_mean_list.append(res["transf_pca_mean_var"][0])
        transf_pca_var_list.append(res["transf_pca_mean_var"][1])
        transf_vector_mean_list.append(res["transf_vector_mean_var"][0])
        transf_vector_var_list.append(res["transf_vector_mean_var"][1])
        transf_random_proj_mean_list.append(res["transf_random_proj_mean_var"][0])
        transf_random_proj_var_list.append(res["transf_random_proj_mean_var"][1])
    
    if dist_measure != "dtw":
        for res in other_results:
            transf_lle_mean_list.append(res["transf_lle_mean_var"][0])
            transf_lle_var_list.append(res["transf_lle_mean_var"][1])
            
            transf_tsne_mean_list.append(res["transf_tsne_mean_var"][0])
            transf_tsne_var_list.append(res["transf_tsne_mean_var"][1])
            
            transf_umap_mean_list.append(res["transf_umap_mean_var"][0])
            transf_umap_var_list.append(res["transf_umap_mean_var"][1])

        mean_transf_pca_mean_list = np.mean(transf_pca_mean_list, axis=0)[:cut_k]
        mean_transf_pca_var_list = np.mean(transf_pca_var_list, axis=0)[:cut_k]
        mean_transf_vector_mean_list = np.mean(transf_vector_mean_list, axis=0)[:cut_k]
        mean_transf_vector_var_list = np.mean(transf_vector_var_list, axis=0)[:cut_k]
        mean_transf_random_proj_mean_list = np.mean(transf_random_proj_mean_list, axis=0)[:cut_k]
        mean_transf_random_proj_var_list = np.mean(transf_random_proj_var_list, axis=0)[:cut_k]
        
        mean_lle_mean_list = np.mean(transf_lle_mean_list, axis = 0)[:cut_k]
        mean_lle_var_list = np.mean(transf_lle_var_list, axis = 0)[:cut_k]
        mean_tsne_mean_list = np.mean(transf_tsne_mean_list, axis = 0)[:cut_k]
        mean_tsne_var_list = np.mean(transf_tsne_var_list, axis = 0)[:cut_k]
        mean_umap_mean_list = np.mean(transf_umap_mean_list, axis = 0)[:cut_k]
        mean_umap_var_list = np.mean(transf_umap_var_list, axis = 0)[:cut_k]
        
    mean_pca_mean_list = np.mean(pca_mean_list, axis=0)[:cut_k]
    mean_pca_var_list = np.mean(pca_var_list, axis=0)[:cut_k]
    mean_vector_mean_list = np.mean(vector_mean_list, axis=0)[:cut_k]
    mean_vector_var_list = np.mean(vector_var_list, axis=0)[:cut_k]
    mean_random_proj_mean_list = np.mean(random_proj_mean_list, axis=0)[:cut_k]
    mean_random_proj_var_list = np.mean(random_proj_var_list, axis=0)[:cut_k]
    
    if dist_measure == "dtw":
        fig, ax1 = plt.subplots(figsize=(14, 10))
    else:
        fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 10))
    
    dims_list = dims_list[:cut_k]
        
    # Plot for PCA for reconstructed data
    upper_var = np.array(mean_pca_mean_list) + np.array(mean_pca_var_list)
    lower_var = np.array(mean_pca_mean_list) - np.array(mean_pca_var_list)

    if sep_measure:
        ax1.plot(dims_list, mean_pca_mean_list, label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.4, color = "#007a33")
    else: 
        ax1.plot(dims_list, mean_pca_mean_list, label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.4, color = "#007a33", linewidth = 2)

    # Plot for random projection for reconstructed data
    upper_var = np.array(mean_random_proj_mean_list) + np.array(mean_random_proj_var_list)
    lower_var = np.array(mean_random_proj_mean_list) - np.array(mean_random_proj_var_list)

    if sep_measure:
        ax1.plot(dims_list, mean_random_proj_mean_list, label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
        
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "#ff42c8")
    else: 
        ax1.plot(dims_list, mean_random_proj_mean_list, label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "#ff42c8", linewidth = 2)
        
    # Plot for Vector PCA for reconstructed data
    upper_var = np.array(mean_vector_mean_list) + np.array(mean_vector_var_list)
    lower_var = np.array(mean_vector_mean_list) - np.array(mean_vector_var_list)

    if sep_measure:
        ax1.plot(dims_list, mean_vector_mean_list, label="PCA*", color = "gold", linewidth=3)
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "gold")
    else: 
        ax1.plot(dims_list, mean_vector_mean_list, label="PCA*", color = "gold", linewidth=3)
        if variance:
            ax1.fill_between(dims_list, lower_var, upper_var, alpha = 0.6, color = "gold", linewidth = 2)

    if dist_measure != "dtw":  
        
        # Plot for lle for transformed data
        upper_var = np.array(mean_lle_mean_list) + np.array(mean_lle_var_list)
        lower_var = np.array(mean_lle_mean_list) - np.array(mean_lle_var_list)

        if sep_measure:
            ax2.plot(dims_list, mean_lle_mean_list, label="LLE", linestyle="dotted", color="#e00031", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3)
        else: 
            ax2.plot(dims_list, mean_lle_mean_list, label="LLE", linestyle="dotted", color="#e00031", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, linewidth = 2)
        
        # Plot for tsne for transformed data
        upper_var = np.array(mean_tsne_mean_list) + np.array(mean_tsne_var_list)
        lower_var = np.array(mean_tsne_mean_list) - np.array(mean_tsne_var_list)

        if sep_measure:
            ax2.plot(dims_list, mean_tsne_mean_list, label="t-SNE", marker=".", color="#9b63f3", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3)
        else: 
            ax2.plot(dims_list, mean_tsne_mean_list, label="t-SNE", marker=".", color="#9b63f3", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, linewidth = 2)
        
        # Plot for umap for transformed data
        upper_var = np.array(mean_umap_mean_list) + np.array(mean_umap_var_list)
        lower_var = np.array(mean_umap_mean_list) - np.array(mean_umap_var_list)

        if sep_measure:
            ax2.plot(dims_list, mean_umap_mean_list, label="UMAP", marker="x", color="#2b9de9", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3)
        else: 
            ax2.plot(dims_list, mean_umap_mean_list, label="UMAP", marker="x", color="#2b9de9", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, linewidth = 2)
            
        # Plot for random projection for transformed data
        upper_var = np.array(mean_transf_random_proj_mean_list) + np.array(mean_transf_random_proj_var_list)
        lower_var = np.array(mean_transf_random_proj_mean_list) - np.array(mean_transf_random_proj_var_list)

        if sep_measure:
            ax2.plot(dims_list, mean_transf_random_proj_mean_list, label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "#ff42c8")
        else: 
            ax2.plot(dims_list, mean_transf_random_proj_mean_list, label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "#ff42c8", linewidth = 2)
        
        # Plot for PCA for transformed data
        upper_var = np.array(mean_transf_pca_mean_list) + np.array(mean_transf_pca_var_list)
        lower_var = np.array(mean_transf_pca_mean_list) - np.array(mean_transf_pca_var_list)

        if sep_measure:
            ax2.plot(dims_list, mean_transf_pca_mean_list, label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "#007a33")
        else: 
            ax2.plot(dims_list, mean_transf_pca_mean_list, label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.6, color = "#007a33", linewidth = 2)

        # Plot for PCA* for transformed data
        upper_var = np.array(mean_transf_vector_mean_list) + np.array(mean_transf_vector_var_list)
        lower_var = np.array(mean_transf_vector_mean_list) - np.array(mean_transf_vector_var_list)

        if sep_measure:
            ax2.plot(dims_list, mean_transf_vector_mean_list, label="PCA*", color = "gold", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.3, color = "gold")
        else: 
            ax2.plot(dims_list, mean_transf_vector_mean_list, label="PCA*", color = "gold", linewidth=3)
            if variance:    
                ax2.fill_between(dims_list, lower_var, upper_var, alpha = 0.6, color = "gold", linewidth = 2)

    if dist_measure != "dtw":
        fontsize = plot_dict["lineplots"]["fontsize"]
        fontdist = plot_dict["lineplots"]["font_dist"]
        fontlegend = plot_dict["lineplots"]["font_legend"]
        if cut_k is None:
            ax1.legend(loc=(1.06, 0.6), fontsize = fontlegend, borderaxespad=0)
            x_steps = 5
        else:
            ax1.legend(loc=(1.06, 0.6), fontsize = fontlegend, borderaxespad=0)
            x_steps = 2
    else:
        fontsize = plot_dict["lineplots"]["dtw_fontsize"]
        fontdist = plot_dict["lineplots"]["dtw_fontdist"]
        fontlegend = plot_dict["lineplots"]["dtw_font_legend"]
        x_steps = 5
        ax1.legend(fontsize = fontlegend, bbox_to_anchor=(1.06, 1))

    ax1.set_xlabel("Dimension " + str(start_dimensions) + " reduced to", fontsize = fontsize - fontdist)
    ax1.set_ylabel("Avg. dev. from \n orig. data", fontsize = fontsize - fontdist)
    ax1.set_title("Results for Reconstruction of " + str(num_runs) + " runs", fontsize = fontsize)
    ax1.tick_params(direction='out', labelsize=fontsize -fontdist)
    ax1.set_axisbelow(True)
    ax1.set_xticks([d for d in dims_list if d % x_steps == 0])
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.yaxis.get_offset_text().set_fontsize(fontsize - fontdist)
    range_max = max(flatten([mean_pca_mean_list, mean_vector_mean_list, mean_random_proj_mean_list]))
    if dist_measure not in ["multiple_scalar_product", "scalar_product"]:
        variance_max = max(flatten([mean_pca_var_list, mean_random_proj_var_list, mean_vector_var_list]))
        range_max += variance_max
    next_higher = pow(10, len(str(round(range_max))))
    if range_max < next_higher - int(next_higher / 2):
            next_higher = next_higher - int(next_higher / 2)
    parts = 5
    ax1.set_yticks(range(0, next_higher, int(next_higher / parts)))
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
        range_max = max(flatten([mean_lle_mean_list, mean_tsne_mean_list, mean_umap_mean_list, mean_transf_pca_mean_list, mean_transf_random_proj_mean_list, mean_transf_vector_mean_list]))
        if dist_measure not in ["multiple_scalar_product", "scalar_product"]:
            variance_max = max(flatten([mean_lle_var_list, mean_tsne_var_list, mean_umap_var_list, mean_transf_pca_var_list, mean_transf_random_proj_var_list, mean_transf_vector_var_list]))
            range_max += variance_max
        if 100 < range_max < 1000:
            lim = 2
        else:
            lim = 0

        ax2.ticklabel_format(style='sci', axis='y', scilimits=(lim, lim))
        ax2.yaxis.get_offset_text().set_fontsize(fontsize - fontdist)
        next_higher = pow(10, len(str(round(range_max))))
        if range_max < next_higher - int(next_higher / 2):
            next_higher = next_higher - int(next_higher / 2)
        if range_max < int(next_higher / 4):
            next_higher = int(next_higher / 4)
        parts = 5
        ax2.set_yticks(range(0, next_higher, int(next_higher / parts)))
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
        plt.savefig("./plots/generated/multiple_runs/" + str(order) + "/avg_dev_vs_dyn_low/" + str(num_lines) + "lines_" + str(num_points) + "points_" + str(k) + "neighbours_" + str(dist_measure) + ".pdf", bbox_inches="tight")
    else:
        plt.savefig("./plots/generated/multiple_runs/" + str(order) + "/avg_dev_vs_dyn_low/" + str(num_lines) + "lines_" + str(num_points) + "points_" + str(k) + "neighbours_" + str(dist_measure) + "_cutk" + str(cut_k) + ".pdf", bbox_inches="tight")
    # ax1.set_yscale("log")
    plt.show()


def multiple_runs_scalar_product(order, start_dim, target_dimensions, num_lines, num_points_per_line, num_runs, bounds, jitter, parallel_start_end, seed = None):
    # Get already computed runs
    already_computed = find_in_pickle_all_without_seed("./pickled_results/scalar_products.pickle", [start_dim, target_dimensions, order, bounds, num_lines, num_points_per_line, jitter, parallel_start_end])

    # Filter seeds of already computed runs
    try:
        already_computed_seeds = [r["parameters"][0] for r in already_computed]
    except:
        already_computed_seeds = []

    print("Already computed seeds: " + str(set(already_computed_seeds)))

    # Sample new seeds which are not already computed
    full_range = list(range(1, 101))
    filtered_range = [num for num in full_range if num not in already_computed_seeds]

    seed_list = random.sample(filtered_range, num_runs)

    if seed is not None:
        seed_list[0] = seed
    print("New seed list " + str(seed_list))

    # Compute scalar products for pca, vector pca, lle and tsne for comparing reconstructed and orgignal, and transformed and original data
    rec_pca_list = []
    rec_vector_pca_list = []

    transf_pca_list = []
    transf_v_pca_list = []

    for seed in seed_list:
        random.seed(seed)
        np.random.seed(seed)
        
        # Generate data
        separated = False

        sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dim, jitter, bounds, parallel_start_end)
        
        vectors = get_neighbor_vector(sep_points, separated)

        if separated:
            data = flatten(sep_points)
        else:
            data = sep_points
        
        # Transform and reconstruct data with pca and vector pca
        pca_rec_data, pca_transf_data = pca_reconstruction(data, target_dimensions, complex_PCA.ComplexPCA, None, None)
        vector_rec_data, vector_transf_data = vector_PCA.vector_pca_reconstruction(data, vectors, target_dimensions)

        scal_prod_rec_pca = scalar_product_measure(data, pca_rec_data)
        scal_prod_transf_pca = scalar_product_measure(data, pca_transf_data)
        scal_prod_rec_vector_pca = scalar_product_measure(data, vector_rec_data)
        scal_prod_transf_vector_pca = scalar_product_measure(data, vector_transf_data)

        # Pickle results
        pickle_results_scalar_product("./pickled_results/scalar_products.pickle", [seed, start_dim, target_dimensions, order, bounds, num_lines, num_points_per_line, jitter, parallel_start_end], scal_prod_rec_pca, scal_prod_transf_pca, scal_prod_rec_vector_pca, scal_prod_transf_vector_pca)
        

def multiple_runs_neighb_error(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, target_dimensions, jitter_bound, num_points, num_lines, max_k, num_runs, scaling_choice = "vector_pca", num_runs_rand = 1):
    assert dist_measure != "scalar_product", "Error: Please use for this function the measure multiple_scalar_product"
    
    if order == "russell1000_stock":
        bounds = [0.08, 2881.94]
        dimensions = 1598
        num_points = 504
    
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    
    slurm_already_computed = find_in_pickle_all_without_seed("./pickled_results/slurm/neighb_error.pickle", [dimensions, target_dimensions, order, bounds, num_lines, num_points, dist_measure, max_k, sep_measure, jitter_bound, parallel_start_end, num_runs, scaling_choice])
    
    slurm_already_computed_other_techn = find_in_pickle_all_without_seed("./pickled_results/slurm/other_methods_neigh_error.pickle", [dimensions, target_dimensions, order, bounds, num_lines, num_points, dist_measure, max_k, sep_measure, jitter_bound, parallel_start_end, num_runs, scaling_choice])
    
    already_computed = find_in_pickle_all_without_seed("./pickled_results/neighb_error.pickle", [dimensions, target_dimensions, order, bounds, num_lines, num_points, dist_measure, max_k, sep_measure, jitter_bound, parallel_start_end, num_runs, scaling_choice])
    
    already_computed_other_techn = find_in_pickle_all_without_seed("./pickled_results/other_methods_neigh_error.pickle", [dimensions, target_dimensions, order, bounds, num_lines, num_points, dist_measure, max_k, sep_measure, jitter_bound, parallel_start_end, num_runs, scaling_choice])
    
    try:
        if len(slurm_already_computed) != 0:
            already_computed = slurm_already_computed
            
        if len(slurm_already_computed_other_techn) != 0:
            already_computed_other_techn = slurm_already_computed_other_techn
    except:
        pass
    
    seed_list = list(range(num_runs))
    if already_computed is not None:
        already_seeds = [r["parameters"][0] for r in already_computed]
    else:
        already_seeds = []
    if already_computed_other_techn is not None:
        already_seeds_oth_techn = [r["parameters"][0] for r in already_computed_other_techn]
    else:
        already_seeds_oth_techn = []
        
    seed_list = [s for s in seed_list if s not in flatten([already_seeds, already_seeds_oth_techn])]
    
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
            rec_data, transf_data = pca_reconstruction(data, target_dimensions, "complex_pca", None, None)
            
            if scaling_choice != "vector_pca":
                complex_data = add_complex_vector(sep_points, separated)
                rec_comp_data, transf_comp_data = pca_reconstruction(complex_data, target_dimensions, complex_PCA.ComplexPCA_svd_complex, scaling_choice,None)
            else:
                neigh_vectors = get_neighbor_vector(sep_points, separated)
                rec_comp_data, transf_comp_data = vector_PCA.vector_pca_reconstruction(data, neigh_vectors, target_dimensions)
            
            transf_random_proj_data, rand_projection = general_func.random_projection(data, target_dimensions)
            reconstr_rand_proj_data = rand_projection.inverse_transform(transf_random_proj_data)
            
            l_reconstr_exp_rand_proj_values = []
            l_reconstr_rand_proj_vars = []
            l_transf_random_proj_exp_values = []
            l_transf_random_proj_vars = []
            
            for r in range(num_runs_rand):
                logging.info("Random projections round " + str(r))
                transf_random_proj_data, rand_projection = general_func.random_projection(data, target_dimensions)
                reconstr_rand_proj_data = rand_projection.inverse_transform(transf_random_proj_data)

                if sep_measure:
                    sep_rec_rand_proj_data = [reconstr_rand_proj_data[i:i + num_points] for i in range(0, len(reconstr_rand_proj_data), num_points)]
                    rec_rand_proj_mean, rec_rand_prj_var, _ = measure_sep_structures(sep_points, sep_rec_rand_proj_data, dist_measure, max_k)
                    
                    if dist_measure != "dtw":
                        sep_transf_random_proj_data = [transf_random_proj_data[i:i + num_points] for i in range(0, len(transf_random_proj_data), num_points)]
                        transf_rand_proj_mean, transf_rand_proj_var, _ = measure_sep_structures(sep_points, sep_transf_random_proj_data, dist_measure, max_k)
                else:
                    rec_rand_proj_mean, rec_rand_prj_var, _ = get_mu_var_by_neighbor_num(data, reconstr_rand_proj_data, dist_measure, max_k)
                    if dist_measure != "dtw":
                        transf_rand_proj_mean, transf_rand_proj_var, _ = get_mu_var_by_neighbor_num(data, transf_random_proj_data, dist_measure, max_k)
                
                l_reconstr_exp_rand_proj_values.append(rec_rand_proj_mean)
                l_reconstr_rand_proj_vars.append(rec_rand_prj_var)
                if dist_measure != "dtw":
                    l_transf_random_proj_exp_values.append(transf_rand_proj_mean)
                    l_transf_random_proj_vars.append(transf_rand_proj_var)
                else:
                    l_transf_random_proj_exp_values.append(0)
                    l_transf_random_proj_vars.append(0)
            reconstr_exp_rand_proj_values = np.mean(l_reconstr_exp_rand_proj_values, axis = 0)
            transf_random_proj_exp_values = np.mean(l_transf_random_proj_exp_values, axis = 0)
            reconstr_rand_proj_vars = np.mean(l_reconstr_rand_proj_vars, axis = 0)
            transf_random_proj_vars = np.mean(l_transf_random_proj_vars, axis = 0)

            # Compute mean and std for original data, complex transformed data, ID data
            if not sep_measure:
                exp_values, vars, median = get_mu_var_by_neighbor_num(data, rec_data, dist_measure, max_k)
                exp_comp_values, comp_vars, comp_median = get_mu_var_by_neighbor_num(data, rec_comp_data.real, dist_measure, max_k)
                
                transf_exp_values, transf_vars, transf_median = get_mu_var_by_neighbor_num(data, transf_data, dist_measure, max_k)
                transf_exp_comp_values, transf_comp_vars, transf_comp_median = get_mu_var_by_neighbor_num(data, transf_comp_data.real, dist_measure, max_k)
                
            elif sep_measure:
                sep_rec_data = [rec_data[i:i + num_points] for i in range(0, len(rec_data), num_points)]
                sep_rec_comp_data = [rec_comp_data[i:i + num_points] for i in range(0, len(rec_comp_data), num_points)]

                exp_values, vars, median = measure_sep_structures(sep_points, sep_rec_data, dist_measure, max_k)
                exp_comp_values, comp_vars, comp_median = measure_sep_structures(sep_points, sep_rec_comp_data, dist_measure, max_k)

                sep_transf_data = [transf_data[i:i + num_points] for i in range(0, len(transf_data), num_points)]
                sep_transf_comp_data = [transf_comp_data[i:i + num_points] for i in range(0, len(transf_comp_data), num_points)]

                transf_exp_values, transf_vars, transf_median = measure_sep_structures(sep_points, sep_transf_data, dist_measure, max_k)
                transf_exp_comp_values, transf_comp_vars, transf_comp_median = measure_sep_structures(sep_points, sep_transf_comp_data, dist_measure, max_k)
                
            pickle_neighbour_error("./pickled_results/neighb_error.pickle", [s, dimensions, target_dimensions, order, bounds, num_lines, num_points, dist_measure, max_k, sep_measure, jitter_bound, parallel_start_end, num_runs, scaling_choice], (exp_values, vars, median), (exp_comp_values, comp_vars, comp_median), (reconstr_exp_rand_proj_values, reconstr_rand_proj_vars, [0]), (transf_exp_values, transf_vars, transf_median), (transf_exp_comp_values, transf_comp_vars, transf_comp_median), (transf_random_proj_exp_values, transf_random_proj_vars, [0]))
            
        if s not in already_seeds_oth_techn:
            logging.info("Other methods: Compute seed " + str(s))
            # Transform data with lle
            try:
                lle = LocallyLinearEmbedding(n_components=target_dimensions)
                lle_transf_data = lle.fit_transform(data)
            except:
                logging.info("Using dense method for lle")
                lle = LocallyLinearEmbedding(n_components=target_dimensions, eigen_solver="dense")
                lle_transf_data = lle.fit_transform(data)

            # Transform data with t-sne
            try:
                tsne = TSNE(n_components=target_dimensions)
                tsne_transf_data = tsne.fit_transform(data)
            except:
                tsne = TSNE(n_components=target_dimensions, method="exact")
                tsne_transf_data = tsne.fit_transform(data)

            # Transform data with umap
            umap_reducer = UMAP(n_components=target_dimensions)
            scaled_data = StandardScaler().fit_transform(data)
            umap_transf_data = umap_reducer.fit_transform(scaled_data)
            
            # Compute mean and std for data transformed by original pca, complex pca and with IDs
            if not sep_measure:
                lle_exp_values, lle_vars, lle_median = get_mu_var_by_neighbor_num(data, lle_transf_data, dist_measure, max_k)
                tsne_exp_values, tsne_vars, tsne_median = get_mu_var_by_neighbor_num(data, tsne_transf_data, dist_measure, max_k)
                umap_exp_values, umap_vars, umap_median = get_mu_var_by_neighbor_num(data, umap_transf_data, dist_measure, max_k)
            elif sep_measure:
                sep_lle_transf_data = [lle_transf_data[i:i + num_points] for i in range(0, len(lle_transf_data), num_points)]
                sep_tsne_transf_data = [tsne_transf_data[i:i + num_points] for i in range(0, len(tsne_transf_data), num_points)]
                sep_umap_transf_data = [umap_transf_data[i:i + num_points] for i in range(0, len(umap_transf_data), num_points)]

                lle_exp_values, lle_vars, lle_median = measure_sep_structures(sep_points, sep_lle_transf_data, dist_measure, max_k)
                tsne_exp_values, tsne_vars, tsne_median = measure_sep_structures(sep_points, sep_tsne_transf_data, dist_measure, max_k)
                umap_exp_values, umap_vars, umap_median = measure_sep_structures(sep_points, sep_umap_transf_data, dist_measure, max_k)

            pickle_neighb_error_other_techniques("./pickled_results/other_methods_neigh_error.pickle", [s, dimensions, target_dimensions, order, bounds, num_lines, num_points, dist_measure, max_k, sep_measure, jitter_bound, parallel_start_end, num_runs, scaling_choice], ( lle_exp_values, lle_vars, lle_median), (tsne_exp_values, tsne_vars, tsne_median), (umap_exp_values, umap_vars, umap_median))
        

def plot_multiple_runs_neighb_error(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, target_dimensions, jitter_bound, num_points, num_lines, max_k, num_runs, variance, cut_k = None, scaling_choice = "vector_pca"):
    assert dist_measure != "dtw", "Error: DTW is not suitable"
    assert dist_measure != "scalar_product", "Error: Please use for this function the measure multiple_scalar_product"
    
    seed_list = list(range(num_runs))
    
    parameters = [dimensions, target_dimensions, order, bounds, num_lines, num_points, dist_measure, max_k, sep_measure, jitter_bound, parallel_start_end, num_runs, scaling_choice]
    result_list = find_in_pickle_all_without_seed("./pickled_results/neighb_error.pickle", parameters)
    slurm_result_list = find_in_pickle_all_without_seed("./pickled_results/slurm/neighb_error.pickle", parameters)
    other_results = find_in_pickle_all_without_seed("./pickled_results/other_methods_neigh_error.pickle", parameters)
    slurm_other_results = find_in_pickle_all_without_seed("./pickled_results/slurm/other_methods_neigh_error.pickle", parameters)
    
    if len(result_list) != num_runs:
        result_list = slurm_result_list
    
    if len(other_results) != num_runs:
        other_results = slurm_other_results
    
    already_seeds = []
    new_results = []
    for i in result_list:
        if i["parameters"] in already_seeds:
            pass
        else:
            new_results.append(i)
    result_list = new_results
    
    already_seeds = []
    new_results = []
    for i in other_results:
        if i["parameters"] in already_seeds:
            pass
        else:
            new_results.append(i)
    other_results = new_results
    
    assert result_list is not None, "Error: There are no pickled results for pca, vector pca and random-projections"
    assert other_results is not None, "Error: There are no pickled results for lle, tsne and umap"
    
    result_list = [entry for entry in result_list if entry["parameters"][0] in seed_list]
    other_results = [entry for entry in other_results if entry["parameters"][0] in seed_list]
    assert len(result_list) == num_runs
    assert len(other_results) == num_runs
    pca_mean_list = []
    pca_var_list = []
    vector_mean_list = []
    vector_var_list = []
    random_proj_mean_list = []
    random_proj_var_list = []

    transf_pca_mean_list = []
    transf_pca_var_list = []
    transf_vector_mean_list = []
    transf_vector_var_list = []
    transf_random_proj_mean_list = []
    transf_random_proj_var_list = []
    
    transf_tsne_mean_list = []
    transf_tsne_var_list = []
    transf_lle_mean_list = []
    transf_lle_var_list = []
    transf_umap_mean_list = []
    transf_umap_var_list = []
    
    for res in result_list:
        pca_mean_list.append(res["pca_results"][0])
        pca_var_list.append(res["pca_results"][1])
        vector_mean_list.append(res["vector_pca_results"][0])
        vector_var_list.append(res["vector_pca_results"][1])
        random_proj_mean_list.append(res["rand_proj_results"][0])
        random_proj_var_list.append(res["rand_proj_results"][1])

        transf_pca_mean_list.append(res["transf_pca_results"][0])
        transf_pca_var_list.append(res["transf_pca_results"][1])
        transf_vector_mean_list.append(res["transf_vector_pca_results"][0])
        transf_vector_var_list.append(res["transf_vector_pca_results"][1])
        transf_random_proj_mean_list.append(res["transf_rand_proj_results"][0])
        transf_random_proj_var_list.append(res["transf_rand_proj_results"][1])
    
    for res in other_results:
        transf_lle_mean_list.append(res["lle_results"][0])
        transf_lle_var_list.append(res["lle_results"][1])
        
        transf_tsne_mean_list.append(res["tsne_results"][0])
        transf_tsne_var_list.append(res["tsne_results"][1])
        
        transf_umap_mean_list.append(res["umap_results"][0])
        transf_umap_var_list.append(res["umap_results"][1])
    
    mean_pca_mean_list = np.mean(pca_mean_list, axis = 0)
    mean_pca_var_list = np.mean(pca_var_list, axis = 0)
    mean_vector_mean_list = np.mean(vector_mean_list, axis = 0)
    mean_vector_var_list = np.mean(vector_var_list, axis = 0)
    mean_random_proj_mean_list = np.mean(random_proj_mean_list, axis = 0)
    mean_random_proj_var_list = np.mean(random_proj_var_list, axis = 0)
    
    mean_transf_pca_mean_list = np.mean(transf_pca_mean_list, axis = 0)
    mean_transf_pca_var_list = np.mean(transf_pca_var_list, axis = 0)
    mean_transf_vector_mean_list = np.mean(transf_vector_mean_list, axis = 0)
    mean_transf_vector_var_list = np.mean(transf_vector_var_list, axis = 0)
    mean_transf_random_proj_mean_list = np.mean(transf_random_proj_mean_list, axis = 0)
    mean_transf_random_proj_var_list = np.mean(transf_random_proj_var_list, axis = 0)
    
    mean_lle_mean_list = np.mean(transf_lle_mean_list, axis = 0)
    mean_lle_var_list = np.mean(transf_lle_var_list, axis = 0)
    mean_tsne_mean_list = np.mean(transf_tsne_mean_list, axis = 0)
    mean_tsne_var_list = np.mean(transf_tsne_var_list, axis = 0)
    mean_umap_mean_list = np.mean(transf_umap_mean_list, axis = 0)
    mean_umap_var_list = np.mean(transf_umap_var_list, axis = 0)
    
    if cut_k is not None:
        max_k = cut_k
        mean_pca_mean_list = mean_pca_mean_list[: max_k]
        mean_pca_var_list = mean_pca_var_list[: max_k]
        mean_vector_mean_list = mean_vector_mean_list[: max_k]
        mean_vector_var_list = mean_vector_var_list[: max_k]
        mean_random_proj_mean_list = mean_random_proj_mean_list[: max_k]
        mean_random_proj_var_list = mean_random_proj_var_list[: max_k]
        
        mean_transf_pca_mean_list = mean_transf_pca_mean_list[: max_k]
        mean_transf_pca_var_list = mean_transf_pca_var_list[: max_k]
        mean_transf_vector_mean_list = mean_transf_vector_mean_list[: max_k]
        mean_transf_vector_var_list = mean_transf_vector_var_list[: max_k]
        mean_transf_random_proj_mean_list = mean_transf_random_proj_mean_list[: max_k]
        mean_transf_random_proj_var_list = mean_transf_random_proj_var_list[: max_k]
        
        mean_lle_mean_list = mean_lle_mean_list[: max_k]
        mean_lle_var_list = mean_lle_var_list[: max_k]
        mean_tsne_mean_list = mean_tsne_mean_list[: max_k]
        mean_tsne_var_list = mean_tsne_var_list[: max_k]
        mean_umap_mean_list = mean_umap_mean_list[: max_k]
        mean_umap_var_list = mean_umap_var_list[: max_k]
    else:
        cut_k = max_k

    fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 10))
    # Plot for PCA for reconstructed data
    upper_var = np.array(mean_pca_mean_list) + np.array(mean_pca_var_list)
    lower_var = np.array(mean_pca_mean_list) - np.array(mean_pca_var_list)
    if sep_measure:
        ax1.plot(range(1, max_k + 1), mean_pca_mean_list, label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
        if variance:
            ax1.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.4, color = "#007a33")
    else: 
        ax1.plot(range(1, max_k + 1), mean_pca_mean_list, label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
        if variance:
            ax1.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.4, color = "#007a33", linewidth = 2)

    # Plot for random projection for reconstructed data
    upper_var = np.array(mean_random_proj_mean_list) + np.array(mean_random_proj_var_list)
    lower_var = np.array(mean_random_proj_mean_list) - np.array(mean_random_proj_var_list)

    if sep_measure:
        ax1.plot(range(1, max_k + 1), mean_random_proj_mean_list, label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
        if variance:    
            ax1.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color = "#ff42c8")
    else: 
        ax1.plot(range(1, max_k + 1), mean_random_proj_mean_list, label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
        if variance:    
            ax1.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color = "#ff42c8", linewidth = 2)
        
    # Plot for Vector PCA for reconstructed data
    upper_var = np.array(mean_vector_mean_list) + np.array(mean_vector_var_list)
    lower_var = np.array(mean_vector_mean_list) - np.array(mean_vector_var_list)

    if sep_measure:
        ax1.plot(range(1, max_k + 1), mean_vector_mean_list, label="PCA*", color = "gold", linewidth=3)
        if variance:
            ax1.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color = "gold")
    else: 
        ax1.plot(range(1, max_k + 1), mean_vector_mean_list, label="PCA*", color = "gold", linewidth=3)
        if variance:    
            ax1.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.6, color = "gold", linewidth = 2)

    
    # Plot for lle for transformed data
    upper_var = np.array(mean_lle_mean_list) + np.array(mean_lle_var_list)
    lower_var = np.array(mean_lle_mean_list) - np.array(mean_lle_var_list)

    if sep_measure:
        ax2.plot(range(1, max_k + 1), mean_lle_mean_list, label="LLE", linestyle="dotted", color="#e00031", linewidth=3)
        if variance:
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color="#e00031")
    else: 
        ax2.plot(range(1, max_k + 1), mean_lle_mean_list, label="LLE", linestyle="dotted", color="#e00031", linewidth=3)
        if variance:
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color="#e00031")
    
    # Plot for tsne for transformed data
    upper_var = np.array(mean_tsne_mean_list) + np.array(mean_tsne_var_list)
    lower_var = np.array(mean_tsne_mean_list) - np.array(mean_tsne_var_list)

    if sep_measure:
        ax2.plot(range(1, max_k + 1), mean_tsne_mean_list, label="t-SNE", marker=".", color="#9b63f3", linewidth=3)
        if variance:
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color="#9b63f3")
    else: 
        ax2.plot(range(1, max_k + 1), mean_tsne_mean_list, label="t-SNE", marker=".", color="#9b63f3", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color="#9b63f3")
    
    # Plot for umap for transformed data
    upper_var = np.array(mean_umap_mean_list) + np.array(mean_umap_var_list)
    lower_var = np.array(mean_umap_mean_list) - np.array(mean_umap_var_list)

    if sep_measure:
        ax2.plot(range(1, max_k + 1), mean_umap_mean_list, label="UMAP", marker="x", color="#2b9de9", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color="#2b9de9")
    else: 
        ax2.plot(range(1, max_k + 1), mean_umap_mean_list, label="UMAP", marker="x", color="#2b9de9", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color="#2b9de9")
        
    # Plot for random projection for transformed data
    upper_var = np.array(mean_transf_random_proj_mean_list) + np.array(mean_transf_random_proj_var_list)
    lower_var = np.array(mean_transf_random_proj_mean_list) - np.array(mean_transf_random_proj_var_list)

    if sep_measure:
        ax2.plot(range(1, max_k + 1), mean_transf_random_proj_mean_list, label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color = "#ff42c8")
    else: 
        ax2.plot(range(1, max_k + 1), mean_transf_random_proj_mean_list, label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color = "#ff42c8", linewidth = 2)
    
    # Plot for PCA for transformed data
    upper_var = np.array(mean_transf_pca_mean_list) + np.array(mean_transf_pca_var_list)
    lower_var = np.array(mean_transf_pca_mean_list) - np.array(mean_transf_pca_var_list)

    if sep_measure:
        ax2.plot(range(1, max_k + 1), mean_transf_pca_mean_list, label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color = "#007a33")
    else: 
        ax2.plot(range(1, max_k + 1), mean_transf_pca_mean_list, label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.6, color = "#007a33", linewidth = 2)

    # Plot for PCA* for transformed data
    upper_var = np.array(mean_transf_vector_mean_list) + np.array(mean_transf_vector_var_list)
    lower_var = np.array(mean_transf_vector_mean_list) - np.array(mean_transf_vector_var_list)

    if sep_measure:
        ax2.plot(range(1, max_k + 1), mean_transf_vector_mean_list, label="PCA*", color = "gold", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color = "gold")
    else: 
        ax2.plot(range(1, max_k + 1), mean_transf_vector_mean_list, label="PCA*", color = "gold", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.6, color = "gold", linewidth = 2)
    fontsize = plot_dict["lineplots"]["fontsize"]
    fontdist = plot_dict["lineplots"]["font_dist"]
    ax1.set_xlabel("Number of neighbours", fontsize = fontsize - fontdist)
    ax1.set_ylabel("Avg. dev. from \n original data", fontsize = fontsize - fontdist)
    ax1.set_title("Results for Reconstruction of 10 runs", fontdict = {"fontsize" : fontsize})
    ax1.legend(loc= "upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0, fontsize = fontsize - fontdist)
    ax1.tick_params(direction='out', labelsize=fontsize - fontdist)
    range_max = max(flatten([mean_pca_mean_list, mean_vector_mean_list, mean_random_proj_mean_list]))
    if dist_measure not in ["multiple_scalar_product", "scalar_product"]:
        variance_max = max(flatten([mean_pca_var_list, mean_random_proj_var_list, mean_vector_var_list]))
        range_max += variance_max
    if order in ["one_line"] and dist_measure in ["multiple_scalar_product", "scalar_product"]:
        range_max += 200
    next_higher = pow(10, len(str(round(range_max))))
    lim = 0
    if range_max < int(next_higher / 2):
            next_higher = int(next_higher / 2)
            lim = 2
    if range_max < int(next_higher / 4):
        next_higher = int(next_higher / 4)
    parts = 5
    ax1.set_yticks(range(0, next_higher, int(next_higher / parts)))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(lim, lim))
    ax1.yaxis.get_offset_text().set_fontsize(fontsize - fontdist)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.grid(axis='y', linestyle="dashed", color="silver", linewidth=1.5)
    
    ax2.set_xlabel("Number of neighbours", fontsize = fontsize - fontdist)
    ax2.set_ylabel("Avg. dev. from \n original data", fontsize = fontsize - fontdist)
    ax2.set_title("Results for Transformation of 10 runs", fontdict = {"fontsize" : fontsize})
    ax2.legend(loc= "upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0, fontsize = fontsize - fontdist)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax2.yaxis.get_offset_text().set_fontsize(fontsize - fontdist)
    ax2.tick_params(direction='out', labelsize=fontsize - fontdist)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.grid(axis='y', linestyle="dashed", color="silver", linewidth=1.5)
    range_max = max(flatten([mean_transf_pca_mean_list, mean_transf_vector_mean_list, mean_transf_random_proj_mean_list, mean_lle_mean_list, mean_tsne_mean_list, mean_umap_mean_list]))
    if dist_measure not in ["multiple_scalar_product", "scalar_product"]:
        variance_max = max(flatten([mean_lle_var_list, mean_tsne_var_list, mean_umap_var_list, mean_transf_pca_var_list, mean_transf_random_proj_var_list, mean_transf_vector_var_list]))
        range_max += variance_max
    if 100 < range_max < 1000:
        lim = 2
    else:
        lim = 0
    next_higher = pow(10, len(str(round(range_max))))
    if range_max < next_higher - int(next_higher / 2):
        next_higher = next_higher - int(next_higher / 2)
    if range_max < int(next_higher / 4):
        next_higher = int(next_higher / 4)
    parts = 5
    ax2.set_yticks(range(0, next_higher, int(next_higher / parts)))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(lim, lim))
    # plt.suptitle("Order " + order + " measure " + str(k) + " neighbours for " + str(num_lines) + " lines a " + str(num_points) + " and jitter " + str(jitter_bound))
    fig.tight_layout(h_pad = 4)
    # fig.set_figwidth(14)
    # fig.set_figheight(8)
    # plt.box(False)
    if order in ["zigzag", "one_line", "sep_lines"]:
        plt.savefig("./plots/generated/multiple_runs/" + str(order) + "/neighb_error/" + str(num_lines) + "lines_" + str(num_points) + "points_" + str(cut_k) + "neighbours_" + str(dist_measure) + ".pdf", bbox_inches="tight")
    
    plt.show()


def compute_multiple_num_neigh_vs_dyn_low_dims(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, jitter_bound, num_points, num_lines, k, num_runs, num_rand_runs = 1):
    
    assert dist_measure not in ["dtw", "scalar_product"], "Error: " + str(dist_measure) + " is not suitable"
    
    if num_runs > 1:
        assert num_rand_runs in [None, 1], "Only run random projection multiple times when you want to compute 1 seed"
        
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    
    low_dim_list = range(2, 11)
    
    seed_list = list(range(num_runs))

    # already_low_pickled = find_in_pickle_all_without_seed("./pickled_results/all_methods_num_neigh_vs_dyn_low_dims.pickle", [dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, low_dim_list])
    
    # slurm_already_low_pickled = find_in_pickle_all_without_seed("./pickled_results/slurm/all_methods_num_neigh_vs_dyn_low_dims.pickle", [dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, low_dim_list])
    
    already_low_pickled = find_in_pickle_all_without_seed("./pickled_results/num_neigh_vs_dyn_low_dims.pickle", [dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, low_dim_list])
    
    slurm_already_low_pickled = find_in_pickle_all_without_seed("./pickled_results/slurm/num_neigh_vs_dyn_low_dims.pickle", [dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, low_dim_list])
    
    other_methods_low_pickled = find_in_pickle_all_without_seed("./pickled_results/other_methods_num_neigh_vs_dyn_low_dims.pickle", [dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, low_dim_list])
    
    slurm_other_methods_low_pickled = find_in_pickle_all_without_seed("./pickled_results/slurm/other_methods_num_neigh_vs_dyn_low_dims.pickle", [dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, low_dim_list])

    if already_low_pickled is None:
        already_low_pickled = slurm_already_low_pickled
        
    if other_methods_low_pickled is None:
        other_methods_low_pickled = slurm_other_methods_low_pickled
    
    try:
        already_pickled_low_seeds = [r["parameters"][0] for r in already_low_pickled]
    except:
        already_pickled_low_seeds = []
        
    try:
        already_pickled_other_low_seeds = [r["parameters"][0] for r in other_methods_low_pickled]
    except:
        already_pickled_other_low_seeds = []
    
    already_seed_list = list(set(already_pickled_other_low_seeds) & set(already_pickled_low_seeds))

    for s in seed_list:
        sep_points, separated = generate_correct_ordered_data(order, num_points, num_lines, dimensions, jitter_bound, bounds, parallel_start_end)
        # Compute neighbours
        vectors = get_neighbor_vector(sep_points, separated)
        
        if s not in already_seed_list:
            logging.info("Computing for seed " + str(s))
            # compute_dyn_low_dim_vs_num_neigh(s, sep_points, separated, vectors, dimensions, k, order, num_lines, num_points, bounds, dist_measure, jitter_bound, parallel_start_end, sep_measure, num_rand_runs)
            compute_dyn_low_dim_vs_num_neigh_alternative(s, sep_points, separated, vectors, dimensions, k, order, num_lines, num_points, bounds, dist_measure, jitter_bound, parallel_start_end, sep_measure, num_rand_runs)
            compute_dyn_low_vs_num_neigh_other_methods(s, sep_points, separated, dimensions, k, order, num_lines, num_points, bounds, dist_measure, jitter_bound, parallel_start_end, sep_measure)
        else: 
            logging.info("Already computed results for seed " + str(s))
            

def compute_dyn_low_dim_vs_num_neigh(seed, sep_points, separated, vectors, start_dim, k, order, num_lines, num_points_per_line, bounds, dist_measure, jitter, parallel_start_end, sep_measure, num_rand_runs = 1):
    dim_list = range(2, 11)
    already_computed_for_order = find_in_pickle_for_specific_data("./pickled_results/all_methods_num_neigh_vs_dyn_low_dims.pickle", order)
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
        list_vector_pca_means_up_to_k = []
        list_rand_proj_means_up_to_k = []
        
        list_transf_pca_means_up_to_k = []
        list_transf_vector_pca_means_up_to_k = []
        list_transf_rand_proj_means_up_to_k = []

        list_lle_means_up_to_k = []
        list_tsne_means_up_to_k = []
        list_umap_means_up_to_k = []
        
        logging.info("Dimensions to be computed " + str(dim_list))
        
        for target_dim in dim_list:
            logging.info("Compute dimension " + str(target_dim))
            logging.info("Compute PCA")
            # Transform and reconstruct data with PCA
            reconstr_data, transf_data = pca_reconstruction(data, target_dim, "complex_pca", None, None)

            # Transform and reconstruct data with Vector PCA
            reconstr_vector_data, transf_vector_data = vector_PCA.vector_pca_reconstruction(data, vectors, target_dim)
            
            logging.info("Compute LLE")
            # Transform with lle
            lle_transf_data = []
            lle_try = 0
            for _ in range(1000):
                logging.info("Try seed: " +  str(lle_try))
                try:
                    lle = LocallyLinearEmbedding(n_components=target_dim, random_state= lle_try)
                    lle_transf_data = lle.fit_transform(data)
                    break
                except:
                    lle_try += 1
                    if lle_try == 99:
                        logging.error("Did not work")
            
            logging.info("Compute tsne")
            # Transform with TSNE
            # assert start_dim < num_lines * num_points_per_line
            tsne_try = 0
            for _ in range(10):
                try:
                    tsne = TSNE(n_components=target_dim, random_state=0)
                    tsne_transf_data = tsne.fit_transform(data)
                    break
                except:
                    tsne_try += 1
                    
            logging.info("Compute umap")
            # Transform with UMAP
            umap_reducer = UMAP(n_components=target_dim)
            scaled_data = StandardScaler().fit_transform(data)
            umap_transf_data = umap_reducer.fit_transform(scaled_data)
            
            # Compute possibly multiple rounds for random projection
            rec_rand_proj_mean_list = []
            transf_rand_proj_mean_list = []
            
            for r in range(num_rand_runs):
                logging.info("Random projection round " + str(r))
                # Transform and reconstruct data with random projection
                transf_random_proj_data, rand_projection = general_func.random_projection(data, target_dim)
                reconstr_rand_proj_data = rand_projection.inverse_transform(transf_random_proj_data)
                if not sep_measure:
                    reconstr_rand_proj_exp_values, _, _ = get_mu_var_by_neighbor_num(data, reconstr_rand_proj_data, dist_measure, k)
                    transf_rand_proj_exp_values, _, _ = get_mu_var_by_neighbor_num(data, transf_random_proj_data, dist_measure, k)
                else:
                    sep_rec_rand_proj_data = [reconstr_rand_proj_data[i:i + num_points_per_line] for i in range(0, len(reconstr_rand_proj_data), num_points_per_line)]
                    reconstr_rand_proj_exp_values, _, _ = measure_sep_structures(sep_points, sep_rec_rand_proj_data, dist_measure, k)
                    
                    sep_transf_rand_proj_data = [transf_random_proj_data[i:i + num_points_per_line] for i in range(0, len(transf_random_proj_data), num_points_per_line)]
                    transf_rand_proj_exp_values, _, _ = measure_sep_structures(sep_points, sep_transf_rand_proj_data, dist_measure, k)
                
                rec_rand_proj_mean_list.append(reconstr_rand_proj_exp_values)
                transf_rand_proj_mean_list.append(transf_rand_proj_exp_values)
                
            reconstr_rand_proj_exp_values = np.mean(rec_rand_proj_mean_list, axis = 0)
            transf_rand_proj_exp_values = np.mean(transf_rand_proj_mean_list, axis = 0)
            
            logging.info("Measure other methods")
            if not sep_measure:
                reconstr_exp_values, _, _ = get_mu_var_by_neighbor_num(data, reconstr_data, dist_measure, k)
                reconstr_vector_exp_values, _, _ = get_mu_var_by_neighbor_num(data, reconstr_vector_data, dist_measure, k)
                
                transf_exp_values, _, _ = get_mu_var_by_neighbor_num(data, transf_data, dist_measure, k)
                transf_vector_exp_values, _, _ = get_mu_var_by_neighbor_num(data, transf_vector_data, dist_measure, k)
                transf_lle_exp_values, _, _ = get_mu_var_by_neighbor_num(data, lle_transf_data, dist_measure, k)
                transf_tsne_exp_values, _, _ = get_mu_var_by_neighbor_num(data, tsne_transf_data, dist_measure, k)
                transf_umap_exp_values, _, _ = get_mu_var_by_neighbor_num(data, umap_transf_data, dist_measure, k)

            elif sep_measure:
                sep_rec_data = [reconstr_data[i:i + num_points_per_line] for i in range(0, len(reconstr_data), num_points_per_line)]
                sep_rec_vector_data = [reconstr_vector_data[i:i + num_points_per_line] for i in range(0, len(reconstr_vector_data), num_points_per_line)]

                reconstr_exp_values, _, _ = measure_sep_structures(sep_points, sep_rec_data, dist_measure, k)
                reconstr_vector_exp_values, _, _ = measure_sep_structures(sep_points, sep_rec_vector_data, dist_measure, k)

                sep_transf_data = [transf_data[i:i + num_points_per_line] for i in range(0, len(transf_data), num_points_per_line)]
                sep_transf_vector_data = [transf_vector_data[i:i + num_points_per_line] for i in range(0, len(transf_vector_data), num_points_per_line)]
                sep_transf_lle_data = [lle_transf_data[i:i + num_points_per_line] for i in range(0, len(lle_transf_data), num_points_per_line)]
                sep_transf_tsne_data = [tsne_transf_data[i:i + num_points_per_line] for i in range(0, len(tsne_transf_data), num_points_per_line)]
                sep_transf_umap_data = [umap_transf_data[i:i + num_points_per_line] for i in range(0, len(umap_transf_data), num_points_per_line)]

                transf_exp_values, _, _ = measure_sep_structures(sep_points, sep_transf_data, dist_measure, k)
                transf_vector_exp_values, _, _ = measure_sep_structures(sep_points, sep_transf_vector_data, dist_measure, k)
                transf_lle_exp_values, _, _ = measure_sep_structures(sep_points, sep_transf_lle_data, dist_measure, k)
                transf_tsne_exp_values, _, _ = measure_sep_structures(sep_points, sep_transf_tsne_data, dist_measure, k)
                transf_umap_exp_values, _, _ = measure_sep_structures(sep_points, sep_transf_umap_data, dist_measure, k)
                
            # Append all
            list_pca_means_up_to_k.append(reconstr_exp_values)
            list_vector_pca_means_up_to_k.append(reconstr_vector_exp_values)
            list_rand_proj_means_up_to_k.append(reconstr_rand_proj_exp_values)
            
            list_transf_pca_means_up_to_k.append(transf_exp_values)
            list_transf_vector_pca_means_up_to_k.append(transf_vector_exp_values)
            list_transf_rand_proj_means_up_to_k.append(transf_rand_proj_exp_values)

            list_lle_means_up_to_k.append(transf_lle_exp_values)
            list_tsne_means_up_to_k.append(transf_tsne_exp_values)
            list_umap_means_up_to_k.append(transf_umap_exp_values)
        logging.info("Finished!")
        pickle_num_neigh_vs_dyn_low_dim("./pickled_results/all_methods_num_neigh_vs_dyn_low_dims.pickle", [seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, k, sep_measure, jitter, parallel_start_end, dim_list], list_pca_means_up_to_k, list_vector_pca_means_up_to_k, list_rand_proj_means_up_to_k, list_transf_pca_means_up_to_k, list_transf_vector_pca_means_up_to_k, list_transf_rand_proj_means_up_to_k, list_lle_means_up_to_k, list_tsne_means_up_to_k, list_umap_means_up_to_k)
    elif parameters in already_computed_parameters:
        logging.info("Parameters already computed for seed " + str(seed))

# As dyn low dim was faster, we try the same way for this
def compute_dyn_low_dim_vs_num_neigh_alternative(seed, sep_points, separated, vectors, start_dim, max_k, order, num_lines, num_points_per_line, bounds, dist_measure, jitter, parallel_start_end, sep_measure, num_rand_runs = 1):
    """
    Compute average deviation vs. dimensionality of the data for dynamically increasing number of dimensions for transformed space.

    Parameters:
        data (array-like): Input data matrix with shape (n_samples, n_features).
        start_dim (int): Number of dimensions specified for original data

    Returns:
        List of means and variances for pca, vector_pca and random_projection
    """
    logging.info("Start computing dynamic dims vs avg dev")
    
    dim_list = range(2, 11)
    parameters_wo_target_dim = [seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, max_k, sep_measure, jitter, parallel_start_end, dim_list]
    
    slurm_pickled = find_in_pickle_dyn_dims_wo_target_dim("./pickled_results/slurm/num_neigh_vs_dyn_low_dims.pickle", parameters_wo_target_dim)
    
    pickled = find_in_pickle_dyn_dims_wo_target_dim("./pickled_results/num_neigh_vs_dyn_low_dims.pickle", parameters_wo_target_dim)

    # Get all already computed target dimensions
    target_dim_list = []
    if slurm_pickled is not None and pickled is not None:
        target_dim_list = [p["parameters"][-1] for p in slurm_pickled] + [p["parameters"][-1] for p in pickled]
    elif slurm_pickled is not None:
        target_dim_list = [p["parameters"][-1] for p in slurm_pickled]
    elif pickled is not None:
        target_dim_list = [p["parameters"][-1] for p in pickled]
        
    # Generate data in current dimensionality
    sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dim, jitter, bounds, parallel_start_end = False, standardise = False)
    if separated:
        data = flatten(sep_points)
    else:
        data = sep_points
    
    for target_dim in dim_list:
        if target_dim in target_dim_list:
            logging.info("Already computed target dim " + str(target_dim))    
        elif target_dim not in target_dim_list:
            logging.info("No pickle found for target dim " + str(target_dim))

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
            
            for r in range(num_rand_runs):
                logging.info("Random projections round " + str(r))
                transf_random_proj_data, rand_projection = general_func.random_projection(data, target_dim)
                reconstr_rand_proj_data = rand_projection.inverse_transform(transf_random_proj_data)

                if sep_measure:
                    sep_rec_rand_proj_data = [reconstr_rand_proj_data[i:i + num_points_per_line] for i in range(0, len(reconstr_rand_proj_data), num_points_per_line)]
                    rec_rand_proj_mean, rec_rand_prj_var, _ = measure_sep_structures(sep_points, sep_rec_rand_proj_data, dist_measure, max_k)
                    
                    if dist_measure != "dtw":
                        sep_transf_random_proj_data = [transf_random_proj_data[i:i + num_points_per_line] for i in range(0, len(transf_random_proj_data), num_points_per_line)]
                        transf_rand_proj_mean, transf_rand_proj_var, _ = measure_sep_structures(sep_points, sep_transf_random_proj_data, dist_measure, max_k)
                else:
                    rec_rand_proj_mean, rec_rand_prj_var, _ = get_mu_var_by_neighbor_num(data, reconstr_rand_proj_data, dist_measure, max_k)
                    if dist_measure != "dtw":
                        transf_rand_proj_mean, transf_rand_proj_var, _ = get_mu_var_by_neighbor_num(data, transf_random_proj_data, dist_measure, max_k)
                
                l_reconstr_exp_rand_proj_values.append(rec_rand_proj_mean)
                l_reconstr_rand_proj_vars.append(rec_rand_prj_var)
                if dist_measure != "dtw":
                    l_transf_random_proj_exp_values.append(transf_rand_proj_mean)
                    l_transf_random_proj_vars.append(transf_rand_proj_var)
                else:
                    l_transf_random_proj_exp_values.append(0)
                    l_transf_random_proj_vars.append(0)
            reconstr_exp_rand_proj_values = np.mean(l_reconstr_exp_rand_proj_values, axis = 0)
            transf_random_proj_exp_values = np.mean(l_transf_random_proj_exp_values, axis = 0)
            reconstr_rand_proj_vars = np.mean(l_reconstr_rand_proj_vars, axis = 0)
            transf_random_proj_vars = np.mean(l_transf_random_proj_vars, axis = 0)

            # Compute exp-values and variances for reconstructed and transformed data
            logging.info("Compute expected values, variances and median")
            if not sep_measure:
                reconstr_exp_values, reconstr_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_data, dist_measure, max_k)
                reconstr_exp_comp_values, reconstr_comp_vars, _ = get_mu_var_by_neighbor_num(data, reconstr_comp_data, dist_measure, max_k)

                if dist_measure != "dtw":
                    transf_exp_values, transf_vars, _ = get_mu_var_by_neighbor_num(data, transf_data, dist_measure, max_k)
                    transf_exp_comp_values, transf_comp_vars, _ = get_mu_var_by_neighbor_num(data, transf_comp_data, dist_measure, max_k)

            elif sep_measure:
                sep_rec_data = [reconstr_data[i:i + num_points_per_line] for i in range(0, len(reconstr_data), num_points_per_line)]
                sep_rec_comp_data = [reconstr_comp_data[i:i + num_points_per_line] for i in range(0, len(reconstr_comp_data), num_points_per_line)]

                reconstr_exp_values, reconstr_vars, _ = measure_sep_structures(sep_points, sep_rec_data, dist_measure, max_k)
                reconstr_exp_comp_values, reconstr_comp_vars, _ = measure_sep_structures(sep_points, sep_rec_comp_data, dist_measure, max_k)

                if dist_measure != "dtw":
                    sep_transf_data = [transf_data[i:i + num_points_per_line] for i in range(0, len(transf_data), num_points_per_line)]
                    sep_transf_comp_data = [transf_comp_data[i:i + num_points_per_line] for i in range(0, len(transf_comp_data), num_points_per_line)]

                    transf_exp_values, transf_vars, _ = measure_sep_structures(sep_points, sep_transf_data, dist_measure, max_k)
                    transf_exp_comp_values, transf_comp_vars, _ = measure_sep_structures(sep_points, sep_transf_comp_data, dist_measure, max_k)

            logging.info("Logging results")
            pickle_results_avg_dev_vs_dyn_dim("./pickled_results/num_neigh_vs_dyn_low_dims.pickle", parameters_wo_target_dim + [target_dim], (reconstr_exp_values, reconstr_vars), (reconstr_exp_comp_values, reconstr_comp_vars), (reconstr_exp_rand_proj_values, reconstr_rand_proj_vars),(transf_exp_values, transf_vars), (transf_exp_comp_values, transf_comp_vars), (transf_random_proj_exp_values, transf_random_proj_vars))
    
    logging.info("Finished computing")
        
        # return[(list_pca_means, list_pca_vars), (list_vector_means, list_vector_vars), (list_rand_proj_means, list_rand_proj_vars),(list_transf_pca_means, list_transf_pca_vars), (list_transf_vector_means, list_transf_vector_vars), (list_transf_random_proj_means, list_transf_random_proj_vars)]


def compute_dyn_low_vs_num_neigh_other_methods(seed, sep_points, separated, start_dim, max_k, order, num_lines, num_points_per_line, bounds, dist_measure, jitter, parallel_start_end, sep_measure):
    """
    Compute average deviation vs. dimensionality of the data for dynamically increasing number of dimensions for transformed space.

    Parameters:
        data (array-like): Input data matrix with shape (n_samples, n_features).
        start_dim (int): Number of dimensions specified for original data

    Returns:
        List of means and variances for pca, vector_pca and random_projection
    """
    dim_list = range(2, 11)
    parameters_wo_target_dim = [seed, start_dim, order, bounds, num_lines, num_points_per_line, dist_measure, max_k, sep_measure, jitter, parallel_start_end, dim_list]
    
    pickled_lle_transf = find_lle_tsne_umap_in_pickle_wo_target_dim("./pickled_results/other_methods_num_neigh_vs_dyn_low_dims.pickle", parameters_wo_target_dim)
    
    slurm_pickled_lle_transf = find_lle_tsne_umap_in_pickle_wo_target_dim("./pickled_results/slurm/other_methods_num_neigh_vs_dyn_low_dims.pickle", parameters_wo_target_dim)
    
    # Get all already computed target dimensions
    target_dim_list = []
    if slurm_pickled_lle_transf is not None and pickled_lle_transf is not None:
        target_dim_list = [p["parameters"][-1] for p in slurm_pickled_lle_transf] + [p["parameters"][-1] for p in pickled_lle_transf]
    elif slurm_pickled_lle_transf is not None:
        target_dim_list = [p["parameters"][-1] for p in slurm_pickled_lle_transf]
    elif pickled_lle_transf is not None:
        target_dim_list = [p["parameters"][-1] for p in pickled_lle_transf]

    separated = False

    sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dim, jitter, bounds, parallel_start_end)
        
    if separated:
        data = flatten(sep_points)
    else:
        data = sep_points
        
    for target_dim in dim_list:
        transf_tsne_exp_values = None
        transf_tsne_vars = None
        
        if target_dim in target_dim_list:
            logging.info("Already computed target dim " + str(target_dim))

        elif target_dim not in target_dim_list:
            logging.info("No pickle found for target dim " + str(target_dim))
            # Transform with lle
            try:
                lle = LocallyLinearEmbedding(n_components=target_dim)
                lle_transf_data = lle.fit_transform(data)
            except:
                logging.info("Using exact for LLE")
                lle = LocallyLinearEmbedding(n_components=target_dim, eigen_solver ="dense")
                lle_transf_data = lle.fit_transform(data)

            # Transform with tsne
            if target_dim < 4:
                try:
                    tsne = TSNE(n_components=target_dim)
                    tsne_transf_data = tsne.fit_transform(data)
                except:
                    tsne_transf_data = TSNE(n_components=target_dim, method = "exact").tsne.fit_transform(data)

            # Transform data with umap
            umap_reducer = UMAP(n_components=target_dim)
            scaled_data = StandardScaler().fit_transform(data)
            umap_transf_data = umap_reducer.fit_transform(scaled_data)

            # Transform data
            # Compute exp-values and variances for reconstructed and transformed data
            if not sep_measure:
                transf_lle_exp_values, transf_lle_vars, _ = get_mu_var_by_neighbor_num(data, lle_transf_data, dist_measure, max_k)
                if target_dim < 4 and order == "flights":
                    transf_tsne_exp_values, transf_tsne_vars, _ = get_mu_var_by_neighbor_num(data, tsne_transf_data.real, dist_measure, max_k)
                umap_transf_exp_values, umap_transf_vars, _ = get_mu_var_by_neighbor_num(data, umap_transf_data, dist_measure, max_k)

            elif sep_measure:
                sep_transf_lle_data = [lle_transf_data[i:i + num_points_per_line] for i in range(0, len(lle_transf_data), num_points_per_line)]
                if target_dim < 4 and order == "flights":
                    sep_transf_tsne_data = [tsne_transf_data[i:i + num_points_per_line] for i in range(0, len(tsne_transf_data), num_points_per_line)]
                sep_umap_transf_data = [umap_transf_data[i:i + num_points_per_line] for i in range(0, len(umap_transf_data), num_points_per_line)]

                transf_lle_exp_values, transf_lle_vars, _ = measure_sep_structures(sep_points, sep_transf_lle_data, dist_measure, max_k)
                if target_dim < 4 and order == "flights":
                    transf_tsne_exp_values, transf_tsne_vars, _ = measure_sep_structures(sep_points, sep_transf_tsne_data, dist_measure, max_k)
                umap_transf_exp_values, umap_transf_vars, _ = measure_sep_structures(sep_points, sep_umap_transf_data, dist_measure, max_k)
        
            pickle_lle_tsne_umap_results_avg_dev_vs_dyn_dim("./pickled_results/other_methods_num_neigh_vs_dyn_low_dims.pickle", parameters_wo_target_dim + [target_dim], (transf_lle_exp_values, transf_lle_vars), (transf_tsne_exp_values, transf_tsne_vars), (umap_transf_exp_values, umap_transf_vars))
        
            pickled_lle_transf = find_lle_tsne_umap_in_pickle("./pickled_results/other_methods_num_neigh_vs_dyn_low_dims.pickle", parameters_wo_target_dim + [target_dim])

            assert len(pickled_lle_transf) != 0 
logging.info("Finished computing")
        


def plot_multiple_num_neigh_vs_dyn_low_dims(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, jitter_bound, num_points, num_lines, k, num_runs, only_pcas = False, real = False):
    assert dist_measure != "dtw", "Error: DTW is not suitable"
    assert dist_measure != "scalar_product", "Error: Please use for this function the measure multiple_scalar_product"

    seed_list = list(range(num_runs))
    if order in ["russell2000_stock", "air_pollution", "flights"]:
        dim_list = range(2, 11)
    else:
        dim_list = generate_integer_list_dyn_low(dimensions, num_lines, order)
    
    parameters = [dimensions, order, bounds, num_lines, num_points, dist_measure, k, sep_measure, jitter_bound, parallel_start_end, dim_list]
    if order not in ["flights"]:
        result_list = find_in_pickle_all_without_seed("./pickled_results/all_methods_num_neigh_vs_dyn_low_dims.pickle", parameters)
        slurm_result_list = find_in_pickle_all_without_seed("./pickled_results/slurm/all_methods_num_neigh_vs_dyn_low_dims.pickle", parameters)
        slurm_result_list = find_in_pickle_all_without_seed("./pickled_results/slurm/all_methods_num_neigh_vs_dyn_low_dims.pickle", parameters)

        try:
            if len(slurm_result_list) != 0:
                result_list = slurm_result_list
        except:
            pass 
    else:
        pickled_results = find_in_pickle_all_without_seed("./pickled_results/slurm/num_neigh_vs_dyn_low_dims.pickle", parameters, False)
        pickled_other_result = find_in_pickle_all_without_seed("./pickled_results/slurm/other_methods_num_neigh_vs_dyn_low_dims.pickle", parameters, False)
        
        result_list = [{
            "parameters": [0] + parameters,
            "list_pca_means": [],
            "list_vector_means": [],
            "list_rand_proj_means": [],
            "list_transf_pca_means": [],
            "list_transf_vector_means":[],
            "list_transf_rand_proj_means": [],
            "list_lle_means": [],
            "list_tsne_means": [],
            "list_umap_means": []
        }]
        already = []
        other_already = []
        for i in dim_list:
            for res in pickled_results:
                if i in already:
                    pass
                else:
                    if res["parameters"][-1] == i:
                        result_list[0]["list_pca_means"].append(res["pca_mean_var"][0])
                        result_list[0]["list_vector_means"].append(res["vector_mean_var"][0])
                        result_list[0]["list_rand_proj_means"].append(res["rand_proj_means_vars"][0])
                        result_list[0]["list_transf_pca_means"].append(res["transf_pca_mean_var"][0])
                        result_list[0]["list_transf_vector_means"].append(res["transf_vector_mean_var"][0])
                        result_list[0]["list_transf_rand_proj_means"].append(res["transf_random_proj_mean_var"][0])
                        already.append(i)
            for res in pickled_other_result:
                if i in other_already:
                    pass
                else:
                    if res["parameters"][-1] == i:
                        result_list[0]["list_lle_means"].append(res["transf_lle_mean_var"][0])
                        if i < 4 and order == "flights":
                            result_list[0]["list_tsne_means"].append(res["transf_tsne_mean_var"][0])
                        else:
                            result_list[0]["list_tsne_means"].append(res["transf_tsne_mean_var"][0])
                        result_list[0]["list_umap_means"].append(res["transf_umap_mean_var"][0])
    assert result_list is not None, "Error: There are no pickled results. Please run the multiple computation function"
    seeds = []
    filtered_results = []
    for res in result_list:
        if res["parameters"][0] not in seeds:
            seeds.append(res["parameters"][0])
            filtered_results.append(res)
    result_list = filtered_results
    
    # Get only means
    result_list = [entry for entry in result_list if entry["parameters"][0] in seed_list]
    
    # other_result_list = [entry for entry in other_result_list if entry["parameters"][0] in seed_list]
    assert len(result_list) == num_runs
    
    pca_results = [entry["list_pca_means"] for entry in result_list]
    
    vector_pca_results = [entry["list_vector_means"] for entry in result_list]
    rand_proj_results = [entry["list_rand_proj_means"] for entry in result_list]
    
    transf_pca_results = [entry["list_transf_pca_means"] for entry in result_list]
    transf_vector_pca_results = [entry["list_transf_vector_means"] for entry in result_list]
    transf_rand_proj_results = [entry["list_transf_rand_proj_means"] for entry in result_list]
    
    lle_results = [entry["list_lle_means"] for entry in result_list]
    tsne_results = [entry["list_tsne_means"] for entry in result_list]
    umap_results = [entry["list_umap_means"] for entry in result_list]
    
    sorted_pca_results = [[] for d in dim_list]
    sorted_vector_pca_results = [[] for d in dim_list]
    sorted_rand_proj_results = [[] for d in dim_list]
    sorted_transf_pca_results = [[] for d in dim_list]
    sorted_transf_vector_pca_results = [[] for d in dim_list]
    sorted_transf_rand_proj_results = [[] for d in dim_list]
    sorted_lle_results = [[] for d in dim_list]
    sorted_tsne_results = [[] for d in dim_list]
    sorted_umap_results = [[] for d in dim_list]
    for s in seed_list:
        for d in range(len(dim_list)):
            sorted_pca_results[d].append(pca_results[s][d])
            sorted_vector_pca_results[d].append(vector_pca_results[s][d])
            sorted_rand_proj_results[d].append(rand_proj_results[s][d])
            sorted_transf_pca_results[d].append(transf_pca_results[s][d])
            sorted_transf_vector_pca_results[d].append(transf_vector_pca_results[s][d])
            sorted_transf_rand_proj_results[d].append(transf_rand_proj_results[s][d])
            sorted_lle_results[d].append(lle_results[s][d])
            if d < 2 and order ==" flights":
                sorted_tsne_results[d].append(tsne_results[s][d])
            else:
                sorted_tsne_results[d].append(tsne_results[s][d])
            sorted_umap_results[d].append(umap_results[s][d])
    mean_pca_results = [np.mean(sorted_pca_results[d], axis = 0) for d in range(len(dim_list))]
    mean_vector_pca_results = [np.mean(sorted_vector_pca_results[d], axis = 0) for d in range(len(dim_list))]
    mean_rand_proj_results = [np.mean(sorted_rand_proj_results[d], axis = 0) for d in range(len(dim_list))]
    mean_transf_pca_results = [np.mean(sorted_transf_pca_results[d], axis = 0) for d in range(len(dim_list))]
    mean_transf_vector_pca_results = [np.mean(sorted_transf_vector_pca_results[d], axis = 0) for d in range(len(dim_list))]
    mean_transf_rand_proj_results = [np.mean(sorted_transf_rand_proj_results[d], axis = 0) for d in range(len(dim_list))]
    mean_lle_results = [np.mean(sorted_lle_results[d], axis = 0) for d in range(len(dim_list))]
    if order == "flights":
        mean_tsne_results = [np.mean(sorted_tsne_results[d], axis = 0) for d in range(2)]
    else:
        mean_tsne_results = [np.mean(sorted_tsne_results[d], axis = 0) for d in range(len(dim_list))]
    mean_umap_results = [np.mean(sorted_umap_results[d], axis = 0) for d in range(len(dim_list))]
    
    all_means = {
        "Reconstruction with PCA" : mean_pca_results, 
        "Reconstruction with PCA*" : mean_vector_pca_results, 
        "Reconstruction with Random Projection" : mean_rand_proj_results, 
        "Transformation with PCA" : mean_transf_pca_results, 
        "Transformation with PCA*" : mean_transf_vector_pca_results, 
        "Transformation with Random Projection" : mean_transf_rand_proj_results, 
        "Transformation with LLE" : mean_lle_results, 
        "Transformation with t-SNE" : mean_tsne_results, 
        "Transformation with UMAP" : mean_umap_results
        }
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    
    fontsize = plot_dict["heatmaps"]["fontsize"]
    fontdist = plot_dict["heatmaps"]["font_dist"]
    if only_pcas:
        fontsize = 40
        fontdist = 4
    x_tick_labels = list(range(1, k + 1))
    y_tick_labels = dim_list
    
    if not only_pcas:
        legend_min = np.min(flatten([mean_pca_results, mean_vector_pca_results, mean_rand_proj_results]))
        legend_max = np.max(flatten([mean_pca_results, mean_vector_pca_results, mean_rand_proj_results]))
        
        if order in ["zigzag", "one_line", "sep_lines"]:
            fig, ax = plt.subplots(3, 3, sharex= True, sharey = True, figsize=(12, 16))
        else:
            fig, ax = plt.subplots(3, 3, sharex= True, sharey = True, figsize=(15, 12))
        annotation = False
    elif only_pcas:
        fig, ax = plt.subplots(1, 2, sharex= True, sharey = True, figsize=(14, 7))
        legend_min = np.min(flatten([mean_pca_results, mean_vector_pca_results]))
        legend_max = np.max(flatten([mean_pca_results, mean_vector_pca_results]))
        all_means = {
            "Reconstruction with PCA" : mean_pca_results, 
            "Reconstruction with PCA*" : mean_vector_pca_results
        }
    
    for i, key in enumerate(list(all_means.keys())[:3]):
        if key == "Reconstruction with Random Projection":
            use_key = "Reconstr. with Rand. Proj."  
        else:
            use_key = key.replace("Reconstruction", "Reconstr.")
        sns.heatmap(all_means[key], ax = ax.flat[i], vmin = legend_min, vmax = legend_max, cmap = "mako", square = False, xticklabels = x_tick_labels, yticklabels = y_tick_labels, cbar_kws={"format": formatter})   
        ax.flat[i].set(xticklabels = x_tick_labels, yticklabels = y_tick_labels)
        ax.flat[i].set_title(use_key, fontsize= fontsize)
        ax.flat[i].tick_params(direction='out', labelsize=fontsize - fontdist, rotation = 0)
        if i == 0:
            ax.flat[i].set_ylabel('Num. of \nTransf. Dim.', fontsize=fontsize - fontdist)
        if only_pcas:
            ax.flat[i].set_xlabel('Number of Neighbours', fontsize=fontsize - fontdist)
        cax = ax.flat[i].figure.axes[-1]
        cax.tick_params(labelsize=fontsize - fontdist)
        
    if not only_pcas:
        legend_min = np.min(flatten([mean_transf_pca_results, mean_transf_vector_pca_results, mean_transf_rand_proj_results, mean_lle_results, mean_tsne_results, mean_umap_results]))
        legend_max = np.max(flatten([mean_transf_pca_results, mean_transf_vector_pca_results, mean_transf_rand_proj_results, mean_lle_results, mean_tsne_results, mean_umap_results]))
        
        for i, key in enumerate(list(all_means.keys())[3:]):
            if key == "Transformation with Random Projection":
                use_key = "Transf. with Rand. Proj."
            else:
                use_key = key.replace("Transformation", "Transf.")
            use_key = use_key.replace("Reconstruction", "Reconstr.")
            sns.heatmap(all_means[key], ax = ax.flat[i + 3], vmin = legend_min, vmax = legend_max, cmap = "mako", xticklabels = x_tick_labels, yticklabels = y_tick_labels, cbar_kws={"format": formatter}) 
            ax.flat[i + 3].set_title(use_key, fontsize= fontsize)
            if i % 3 == 0:
                ax.flat[i + 3].set_ylabel('Number of \nTransf. Dim.', fontsize=fontsize - fontdist)
            if i > 2:
                ax.flat[i + 3].set_xlabel('Number of Neighbours', fontsize=fontsize - fontdist)
            ax.flat[i + 3].tick_params(direction='out', labelsize=fontsize - fontdist, rotation = 0)
            cax = ax.flat[i].figure.axes[-1]
            cax.tick_params(labelsize=fontsize - fontdist)
    plt.subplots_adjust(top=0.8)    
    plt.tight_layout()
    if order not in ["russell2000_stock", "flights", "air_pollution"]:
        plt.savefig("./plots/generated/multiple_runs/" + str(order) + "/dyn_low_dim_vs_num_neigh/" + str(dist_measure) + "/all_methods_" + str(num_runs)+ "runs_"  + str(num_lines) + "lines_" + str(num_points) + "points_" + str(k) + "neighbours.pdf", bbox_inches="tight")
    else:
        if not only_pcas:
            plt.savefig("./plots/real-world/" + str(order) + "/dyn_low_dim_vs_num_neigh/" + str(dist_measure) + "/all_methods_" + str(num_runs)+ "runs_"  + str(num_lines) + "lines_" + str(num_points) + "points_" + str(k) + "neighbours.pdf", bbox_inches="tight")
        else:
            plt.savefig("./plots/real-world/" + str(order) + "/dyn_low_dim_vs_num_neigh/" + str(dist_measure) + "/onlypcas" + str(num_runs)+ "runs_"  + str(num_lines) + "lines_" + str(num_points) + "points_" + str(k) + "neighbours.pdf", bbox_inches="tight")
    plt.show()
        
    
def compute_multiple_num_lines_vs_num_neigh(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, jitter_bound, num_points, max_num_lines, max_k, num_runs, target_dim):
    
    assert dist_measure not in ["dtw", "scalar_product"], "Error: " + str(dist_measure) + " is not suitable. Use euclidean or multiiple_scalar_product measure"
    
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    
    seed_list = list(range(num_runs))

    already_low_pickled = find_in_pickle_all_without_seed("./pickled_results/all_methods_num_neigh_vs_num_lines.pickle", [dimensions, order, bounds, max_num_lines, num_points, dist_measure, max_k, sep_measure, jitter_bound, parallel_start_end, target_dim])
    slurm_already_low_pickled = find_in_pickle_all_without_seed("./pickled_results/slurm/all_methods_num_neigh_vs_num_lines.pickle", [dimensions, order, bounds, max_num_lines, num_points, dist_measure, max_k, sep_measure, jitter_bound, parallel_start_end, target_dim])

    if already_low_pickled is None:
        already_low_pickled = slurm_already_low_pickled
    
    try:
        already_pickled_low_seeds = [r["parameters"][0] for r in already_low_pickled]
    except:
        already_pickled_low_seeds = []

    for s in seed_list:
        
        if s not in already_pickled_low_seeds:
            logging.info("Computing for seed " + str(s))
            compute_num_lines_vs_num_neigh(s, dimensions, max_k, order, max_num_lines, num_points, bounds, dist_measure, jitter_bound, parallel_start_end, sep_measure,target_dim)
        else:
            logging.info("Already computed results for seed " + str(s))
            
            
def compute_num_lines_vs_num_neigh(seed, start_dim, max_k, order, max_num_lines, num_points_per_line, bounds, dist_measure, jitter_bound, parallel_start_end, sep_measure, target_dim):
    
    list_pca_means_up_to_k = []
    list_vector_pca_means_up_to_k = []
    list_rand_proj_means_up_to_k = []
    
    list_transf_pca_means_up_to_k = []
    list_transf_vector_pca_means_up_to_k = []
    list_transf_rand_proj_means_up_to_k = []

    list_lle_means_up_to_k = []
    list_tsne_means_up_to_k = []
    list_umap_means_up_to_k = []
    
    for num_lines in range(1, max_num_lines + 1):
        sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dim, jitter_bound, bounds, parallel_start_end)
        if separated:
            data = flatten(sep_points)
        else:
            data = sep_points
            
        # Compute neighbours
        vectors = get_neighbor_vector(sep_points, separated)
            
        # Transform and reconstruct with PCA
        reconstr_data, transf_data = pca_reconstruction(data, target_dim, "complex_pca", None, None)
        
        # Transform and reconstruct data with Vector PCA
        reconstr_vector_data, transf_vector_data = vector_PCA.vector_pca_reconstruction(data, vectors, target_dim)
        
        # Transform and reconstruct data with random projection
        transf_random_proj_data, rand_projection = general_func.random_projection(data, target_dim)
        reconstr_rand_proj_data = rand_projection.inverse_transform(transf_random_proj_data)

        # Transform with lle
        lle = LocallyLinearEmbedding(n_components=target_dim, eigen_solver="dense")
        lle_transf_data = lle.fit_transform(data)
        
        # Transform with TSNE
        assert start_dim <= num_lines * num_points_per_line
        tsne = TSNE(n_components=target_dim, method="exact")
        tsne_transf_data = tsne.fit_transform(data)

        # Transform with UMAP
        umap_reducer = UMAP(n_components=target_dim)
        scaled_data = StandardScaler().fit_transform(data)
        umap_transf_data = umap_reducer.fit_transform(scaled_data)
        
        if not sep_measure:
            reconstr_exp_values, _, _ = get_mu_var_by_neighbor_num(data, reconstr_data, dist_measure, max_k)
            reconstr_vector_exp_values, _, _ = get_mu_var_by_neighbor_num(data, reconstr_vector_data, dist_measure, max_k)
            reconstr_rand_proj_exp_values, _, _ = get_mu_var_by_neighbor_num(data, reconstr_rand_proj_data, dist_measure, max_k)
            
            transf_exp_values, _, _ = get_mu_var_by_neighbor_num(data, transf_data, dist_measure, max_k)
            transf_vector_exp_values, _, _ = get_mu_var_by_neighbor_num(data, transf_vector_data, dist_measure, max_k)
            transf_rand_proj_exp_values, _, _ = get_mu_var_by_neighbor_num(data, transf_random_proj_data, dist_measure, max_k)
            transf_lle_exp_values, _, _ = get_mu_var_by_neighbor_num(data, lle_transf_data, dist_measure, max_k)
            transf_tsne_exp_values, _, _ = get_mu_var_by_neighbor_num(data, tsne_transf_data, dist_measure, max_k)
            transf_umap_exp_values, _, _ = get_mu_var_by_neighbor_num(data, umap_transf_data, dist_measure, max_k)

        elif sep_measure:
            sep_rec_data = [reconstr_data[i:i + num_points_per_line] for i in range(0, len(reconstr_data), num_points_per_line)]
            sep_rec_vector_data = [reconstr_vector_data[i:i + num_points_per_line] for i in range(0, len(reconstr_vector_data), num_points_per_line)]
            sep_rec_rand_proj_data = [reconstr_rand_proj_data[i:i + num_points_per_line] for i in range(0, len(reconstr_rand_proj_data), num_points_per_line)]

            reconstr_exp_values, _, _ = measure_sep_structures(sep_points, sep_rec_data, dist_measure, max_k)
            reconstr_vector_exp_values, _, _ = measure_sep_structures(sep_points, sep_rec_vector_data, dist_measure, max_k)
            reconstr_rand_proj_exp_values, _, _ = measure_sep_structures(sep_points, sep_rec_rand_proj_data, dist_measure, max_k)

            sep_transf_data = [transf_data[i:i + num_points_per_line] for i in range(0, len(transf_data), num_points_per_line)]
            sep_transf_vector_data = [transf_vector_data[i:i + num_points_per_line] for i in range(0, len(transf_vector_data), num_points_per_line)]
            sep_transf_rand_proj_data = [transf_random_proj_data[i:i + num_points_per_line] for i in range(0, len(transf_random_proj_data), num_points_per_line)]
            sep_transf_lle_data = [lle_transf_data[i:i + num_points_per_line] for i in range(0, len(lle_transf_data), num_points_per_line)]
            sep_transf_tsne_data = [tsne_transf_data[i:i + num_points_per_line] for i in range(0, len(tsne_transf_data), num_points_per_line)]
            sep_transf_umap_data = [umap_transf_data[i:i + num_points_per_line] for i in range(0, len(umap_transf_data), num_points_per_line)]

            transf_exp_values, _, _ = measure_sep_structures(sep_points, sep_transf_data, dist_measure, max_k)
            transf_vector_exp_values, _, _ = measure_sep_structures(sep_points, sep_transf_vector_data, dist_measure, max_k)
            transf_rand_proj_exp_values, _, _ = measure_sep_structures(sep_points, sep_transf_rand_proj_data, dist_measure, max_k)
            transf_lle_exp_values, _, _ = measure_sep_structures(sep_points, sep_transf_lle_data, dist_measure, max_k)
            transf_tsne_exp_values, _, _ = measure_sep_structures(sep_points, sep_transf_tsne_data, dist_measure, max_k)
            transf_umap_exp_values, _, _ = measure_sep_structures(sep_points, sep_transf_umap_data, dist_measure, max_k)
            
        # Append all
        list_pca_means_up_to_k.append(reconstr_exp_values)
        list_vector_pca_means_up_to_k.append(reconstr_vector_exp_values)
        list_rand_proj_means_up_to_k.append(reconstr_rand_proj_exp_values)
        
        list_transf_pca_means_up_to_k.append(transf_exp_values)
        list_transf_vector_pca_means_up_to_k.append(transf_vector_exp_values)
        list_transf_rand_proj_means_up_to_k.append(transf_rand_proj_exp_values)

        list_lle_means_up_to_k.append(transf_lle_exp_values)
        list_tsne_means_up_to_k.append(transf_tsne_exp_values)
        list_umap_means_up_to_k.append(transf_umap_exp_values)
        
    pickle_num_neigh_vs_dyn_low_dim("./pickled_results/all_methods_num_neigh_vs_num_lines.pickle", [seed, start_dim, order, bounds, max_num_lines, num_points_per_line, dist_measure, max_k, sep_measure, jitter_bound, parallel_start_end, target_dim], list_pca_means_up_to_k, list_vector_pca_means_up_to_k, list_rand_proj_means_up_to_k, list_transf_pca_means_up_to_k, list_transf_vector_pca_means_up_to_k, list_transf_rand_proj_means_up_to_k, list_lle_means_up_to_k, list_tsne_means_up_to_k, list_umap_means_up_to_k)
    

def plot_num_lines_vs_num_neigh(start_dim, max_k, order, max_num_lines, num_points_per_line, bounds, dist_measure, jitter_bound, parallel_start_end, sep_measure, target_dim, num_runs):
    assert dist_measure != "dtw", "Error: DTW is not suitable"
    assert dist_measure != "scalar_product", "Error: Please use for this function the measure multiple_scalar_product"

    seed_list = list(range(num_runs))
    
    parameters = [start_dim, order, bounds, max_num_lines, num_points_per_line, dist_measure, max_k, sep_measure, jitter_bound, parallel_start_end, target_dim]
    
    result_list = find_in_pickle_all_without_seed("./pickled_results/all_methods_num_neigh_vs_num_lines.pickle", parameters)
    slurm_result_list = find_in_pickle_all_without_seed("./pickled_results/slurm/all_methods_num_neigh_vs_num_lines.pickle", parameters)
    
    if len(slurm_result_list) != 0 :
        result_list = slurm_result_list
        
    seeds = [] 
    new_results = []
    
    for res in result_list:
        if res["parameters"][0] not in seeds:
            seeds.append(res["parameters"][0])
            new_results.append(res)
    result_list = new_results
    # print(slurm_result_list)
    assert result_list is not None, "Error: There are no pickled results. Please run the multiple computation function"
    
    result_list = [entry for entry in result_list if entry["parameters"][0] in seed_list]
    assert len(result_list) == num_runs
    
    pca_results = [entry["list_pca_means"] for entry in result_list]
    vector_pca_results = [entry["list_vector_means"] for entry in result_list]
    rand_proj_results = [entry["list_rand_proj_means"] for entry in result_list]
    
    transf_pca_results = [entry["list_transf_pca_means"] for entry in result_list]
    transf_vector_pca_results = [entry["list_transf_vector_means"] for entry in result_list]
    transf_rand_proj_results = [entry["list_transf_rand_proj_means"] for entry in result_list]
    
    lle_results = [entry["list_lle_means"] for entry in result_list]
    tsne_results = [entry["list_tsne_means"] for entry in result_list]
    umap_results = [entry["list_umap_means"] for entry in result_list]
    
    sorted_pca_results = [[] for d in range(max_num_lines)]
    sorted_vector_pca_results = [[] for d in range(max_num_lines)]
    sorted_rand_proj_results = [[] for d in range(max_num_lines)]
    sorted_transf_pca_results = [[] for d in range(max_num_lines)]
    sorted_transf_vector_pca_results = [[] for d in range(max_num_lines)]
    sorted_transf_rand_proj_results = [[] for d in range(max_num_lines)]
    sorted_lle_results = [[] for d in range(max_num_lines)]
    sorted_tsne_results = [[] for d in range(max_num_lines)]
    sorted_umap_results = [[] for d in range(max_num_lines)]
    
    for s in seed_list:
        for d in range(max_num_lines):
            sorted_pca_results[d].append(pca_results[s][d])
            sorted_vector_pca_results[d].append(vector_pca_results[s][d])
            sorted_rand_proj_results[d].append(rand_proj_results[s][d])
            sorted_transf_pca_results[d].append(transf_pca_results[s][d])
            sorted_transf_vector_pca_results[d].append(transf_vector_pca_results[s][d])
            sorted_transf_rand_proj_results[d].append(transf_rand_proj_results[s][d])
            sorted_lle_results[d].append(lle_results[s][d])
            sorted_tsne_results[d].append(tsne_results[s][d])
            sorted_umap_results[d].append(umap_results[s][d])
    
    mean_pca_results = [np.mean(sorted_pca_results[d], axis = 0) for d in range(max_num_lines)]
    mean_vector_pca_results = [np.mean(sorted_vector_pca_results[d], axis = 0) for d in range(max_num_lines)]
    mean_rand_proj_results = [np.mean(sorted_rand_proj_results[d], axis = 0) for d in range(max_num_lines)]
    mean_transf_pca_results = [np.mean(sorted_transf_pca_results[d], axis = 0) for d in range(max_num_lines)]
    mean_transf_vector_pca_results = [np.mean(sorted_transf_vector_pca_results[d], axis = 0) for d in range(max_num_lines)]
    mean_transf_rand_proj_results = [np.mean(sorted_transf_rand_proj_results[d], axis = 0) for d in range(max_num_lines)]
    mean_lle_results = [np.mean(sorted_lle_results[d], axis = 0) for d in range(max_num_lines)]
    mean_tsne_results = [np.mean(sorted_tsne_results[d], axis = 0) for d in range(max_num_lines)]
    mean_umap_results = [np.mean(sorted_umap_results[d], axis = 0) for d in range(max_num_lines)]
    
    all_means = {
        "Reconstruction with PCA" : mean_pca_results, 
        "Reconstruction with PCA*" : mean_vector_pca_results, 
        "Reconstruction with Random Projection" : mean_rand_proj_results, 
        "Transformation with PCA" : mean_transf_pca_results, 
        "Transformation with PCA*" : mean_transf_vector_pca_results, 
        "Transformation with Random Projection" : mean_transf_rand_proj_results, 
        "Transformation with LLE" : mean_lle_results, 
        "Transformation with t-SNE" : mean_tsne_results, 
        "Transformation with UMAP" : mean_umap_results
        }
    x_tick_labels = list(range(1, max_k + 1))
    y_tick_labels = range(1, max_num_lines + 1)
    
    legend_min = np.min(flatten([mean_pca_results, mean_vector_pca_results, mean_rand_proj_results]))
    legend_max = np.max(flatten([mean_pca_results, mean_vector_pca_results, mean_rand_proj_results]))
    
    fig, ax = plt.subplots(3, 3, sharex= True, sharey = True, figsize=(10, 8))
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    
    fontsize = plot_dict["heatmaps"]["fontsize"]
    fontdist = plot_dict["heatmaps"]["font_dist"]
    
    for i, key in enumerate(list(all_means.keys())[:3]):
        if key == "Reconstruction with Random Projection":
            use_key = "Reconstr. with Rand. Proj."  
        else:
            use_key = key
        if key in ["Transformation with LLE", "Transformation with t-SNE", "Transformation with UMAP"]:
            break
        sns.heatmap(all_means[key], ax = ax.flat[i], vmin = legend_min, vmax = legend_max, cmap = "mako", square = False, xticklabels = x_tick_labels, yticklabels = y_tick_labels, cbar_kws={"format": formatter})
        ax.flat[i].set_title(use_key, fontsize= fontsize)
        # ax.flat[i].set(xticklabels = x_tick_labels, yticklabels = y_tick_labels)
        ax.flat[i].tick_params(direction='out', labelsize=fontsize - fontdist, rotation = 0)
        if i == 0:
            ax.flat[i].set_ylabel('Num. of Lines', fontsize=fontsize - fontdist)
        cax = ax.flat[i].figure.axes[-1]
        cax.tick_params(labelsize=16)
    
    legend_min = np.min(flatten([mean_transf_pca_results, mean_transf_vector_pca_results, mean_transf_rand_proj_results, mean_lle_results, mean_tsne_results, mean_umap_results]))
    legend_max = np.max(flatten([mean_transf_pca_results, mean_transf_vector_pca_results, mean_transf_rand_proj_results, mean_lle_results, mean_tsne_results, mean_umap_results]))
    
    for i, key in enumerate(list(all_means.keys())[3:]):
        if key == "Transformation with Random Projection":
            use_key = "Transf. with Rand. Proj."
        else:
            use_key = key.replace("Transformation", "Transf.")
        sns.heatmap(all_means[key], ax = ax.flat[i + 3], vmin = legend_min, vmax = legend_max, cmap = "mako", xticklabels = x_tick_labels, yticklabels = y_tick_labels, cbar_kws={"format": formatter})
        ax.flat[i + 3].set_title(use_key, fontsize= fontsize)
        if i == 0 or i == 3:
            ax.flat[i + 3].set_ylabel('Num. of Lines', fontsize=fontsize - fontdist)
        if i > 2:
            ax.flat[i + 3].set_xlabel('Number of neighbours', fontsize=fontsize - fontdist)
        ax.flat[i + 3].tick_params(direction='out', labelsize=fontsize - fontdist, rotation = 0)
        cax = ax.flat[i].figure.axes[-1]
        cax.tick_params(labelsize=fontsize - fontdist)
    
    plt.tight_layout()
    plt.savefig("./plots/generated/multiple_runs/" + str(order) + "/num_lines_vs_num_neigh/" + str(dist_measure) + "/all_methods_" + str(num_runs)+ "runs_"  + str(max_num_lines) + "lines_" + str(num_points_per_line) + "points_" + str(max_k) + "neighbours.pdf", bbox_inches="tight")
    plt.show()
        
    
        
    