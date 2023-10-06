from structure_measures import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import general_func
import complex_PCA
import vector_PCA
from general_func import *
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import LocallyLinearEmbedding, TSNE


def heatmap_dyn_red_dyns_vs_lines(seed, start_dim, k_neigh, order, max_num_lines, num_points_per_line, bounds, dist_measure, jitter, parallel_start_end, sep_measure):
    '''
    Generates data in increasing number of lines with start_dim dimensions and transforms it to increasing dimensions.

    Returns:
    Heatmap quality of reconstruction and transformation in number of lines vs. transformation dimensions
    '''
    
    pickled = find_in_pickle_lines_vs_dyn_dim("./pickled_results/dyn_red_dyns_vs_lines.pickle", [seed, start_dim, k_neigh, order, max_num_lines, num_points_per_line, bounds, dist_measure, jitter, parallel_start_end, sep_measure])

    # list = [lines] => lines = [dim`s] 
    # list = [[dims`s], [dim's],...]
    #          line 1    line2 ... 
    list_pca_means = []
    list_vector_pca_means = []
    list_transf_pca_means = []
    list_transf_vector_pca_means = []
    dim_list = general_func.generate_integer_list_dyn_low(start_dim, max_num_lines, order)
    separated = False

    if pickled is not None:
        list_pca_means = pickled[0]
        list_vector_pca_means = pickled[1]
        list_transf_pca_means = pickled[2]
        list_transf_vector_pca_means = pickled[3]

    elif pickled is None:

        for num_lines in range(1, max_num_lines +1):
            # Generate data in current dimensionality
            if order == "staircase":
                separated = True
                data = generate_staircase_lines(num_lines, num_points_per_line, bounds, start_dim, rotation_deg = 0)
                sep_points = data
            elif order == "connected_staircase":
                sep_points = flatten(generate_staircase_lines(num_lines, num_points_per_line, bounds, start_dim, rotation_deg = 0))
                data = sep_points
            elif order == "parallel_lines":
                sep_points = generate_parallel_lines(num_lines, num_points_per_line, start_dim, bounds, 0, jitter, parallel_start_end=parallel_start_end)
                data = sep_points
                separated = True
            elif order == "connected_parallel_lines":
                sep_points = flatten(generate_parallel_lines(num_lines, num_points_per_line, start_dim, bounds, 0, jitter, parallel_start_end=parallel_start_end))
                data = sep_points
            elif order == "connected_zigzag":
                data = generate_connected_zigzag(num_points_per_line, num_lines, start_dim, bounds, jitter)
                sep_points = data
            elif order == "clusters":
                data = generate_clusters(num_lines, num_points_per_line, 1, bounds, start_dim)
                separated = True
                sep_points = data
            else:
                sep_points = generate_data(bounds, start_dim, num_points_per_line, num_lines, jitter)
                data = orderings.order_nearest_start_points(sep_points)
                separated = True

                if order == "zigzag":
                    data = orderings.zigzag_order(data, num_points_per_line)
                    separated = False
                    sep_points = data
                elif order == "one_line":
                    data = orderings.one_line(data)
                    separated = False
                    sep_points = data
                elif order == "random":
                    data = orderings.random_ordering(flatten(data))
                    separated = False
                    sep_points = data
                elif order == "sep_lines":
                    separated = True
                    sep_points = data

            if separated:
                data = flatten(sep_points)
            
            line_dim_result = []
            vector_line_dim_result = []

            transf_line_dim_result = []
            transf_vector_line_dim_result = []

            for target_dim in dim_list:

                # Transform  and reconstruct data
                reconstr_data, transf_data = pca_reconstruction(data, target_dim, complex_PCA.ComplexPCA, None, None)

                # Reconstruct vector data
                vectors = get_neighbor_vector(sep_points, separated)
                reconstr_comp_data, transf_comp_data = vector_PCA.vector_pca_reconstruction(data, vectors, target_dim)

                # Compute exp-values and variances for reconstructed and transformed data
                if not sep_measure:
                    reconstr_exp_values, _, _ = get_mu_var_by_neighbor_num(data, reconstr_data, dist_measure, k_neigh)
                    reconstr_exp_comp_values, _, _ = get_mu_var_by_neighbor_num(data, reconstr_comp_data, dist_measure, k_neigh)

                    transf_exp_values, _, _ = get_mu_var_by_neighbor_num(data, transf_data, dist_measure, k_neigh)
                    transf_exp_comp_values, _, _ = get_mu_var_by_neighbor_num(data, transf_comp_data, dist_measure, k_neigh)

                elif sep_measure:
                    sep_rec_data = [reconstr_data[i:i + num_points_per_line] for i in range(0, len(reconstr_data), num_points_per_line)]
                    sep_rec_comp_data = [reconstr_comp_data[i:i + num_points_per_line] for i in range(0, len(reconstr_comp_data), num_points_per_line)]

                    reconstr_exp_values, _, _ = measure_sep_structures(sep_points, sep_rec_data, dist_measure, k_neigh)
                    reconstr_exp_comp_values, _, _ = measure_sep_structures(sep_points, sep_rec_comp_data, dist_measure, k_neigh)

                    sep_transf_data = [transf_data[i:i + num_points_per_line] for i in range(0, len(transf_data), num_points_per_line)]
                    sep_transf_comp_data = [transf_comp_data[i:i + num_points_per_line] for i in range(0, len(transf_comp_data), num_points_per_line)]

                    transf_exp_values, _, _ = measure_sep_structures(sep_points, sep_transf_data, dist_measure, k_neigh)
                    transf_exp_comp_values, _, _ = measure_sep_structures(sep_points, sep_transf_comp_data, dist_measure, k_neigh)

                line_dim_result.append(reconstr_exp_values[-1])
                vector_line_dim_result.append(reconstr_exp_comp_values[-1])

                transf_line_dim_result.append(transf_exp_values[-1])
                transf_vector_line_dim_result.append(transf_exp_comp_values[-1])
            
            list_pca_means.append(line_dim_result)
            list_vector_pca_means.append(vector_line_dim_result)
            list_transf_pca_means.append(transf_line_dim_result)
            list_transf_vector_pca_means.append(transf_vector_line_dim_result)

        pickle_results_lines_vs_dyn_dim("./pickled_results/dyn_red_dyns_vs_lines.pickle", [seed, start_dim, k_neigh, order, max_num_lines, num_points_per_line, bounds, dist_measure, jitter, parallel_start_end, sep_measure], list_pca_means, list_vector_pca_means, list_transf_pca_means, list_transf_vector_pca_means)
        
    fig, axes = plt.subplots(2,2)

    rec_pca = pd.DataFrame(list_pca_means, columns=dim_list, index = list(range(1,max_num_lines + 1)))
    sns.heatmap(rec_pca, ax = axes[0, 0], annot=False)
    axes[0, 0].set_title("PCA")
    axes[0, 0].set(xlabel="Transformation dimensionality", ylabel="Number of lines")

    rec_vector_pca = pd.DataFrame(list_vector_pca_means, columns=dim_list, index = list(range(1,max_num_lines + 1)))
    sns.heatmap(data=rec_vector_pca, ax = axes[0, 1])    
    axes[0, 1].set_title("Vector PCA")
    axes[0, 1].set(xlabel="Transformation dimensionality", ylabel="Number of lines")

    transf_pca = pd.DataFrame(list_transf_pca_means, columns=dim_list, index = list(range(1,max_num_lines + 1)))
    sns.heatmap(transf_pca, ax = axes[1, 0])   
    axes[1, 0].set_title("Transformed: PCA")
    axes[1, 0].set(xlabel="Transformation dimensionality", ylabel="Number of lines")

    transf_vector_pca = pd.DataFrame(list_transf_vector_pca_means, columns=dim_list, index = list(range(1,max_num_lines + 1)))
    sns.heatmap(transf_vector_pca, ax = axes[1, 1])   
    axes[1, 1].set_title("Transformed: Vector PCA")
    axes[1, 1].set(xlabel="Transformation dimensionality", ylabel="Number of lines")

    fig.suptitle("Original space of dimensionality " + str(start_dim) + " in Order \"" + str(order) + "\" with jitter " + str(jitter) + "\nNumber of points per line: " + str(num_points_per_line) + " and observed neighbours "+ str(k_neigh))
    plt.tight_layout()
    plt.show()
    

def heatmap_dyn_red_dyns_vs_neighbours(seed, start_dim, k_neigh, order, num_lines, num_points_per_line, bounds, dist_measure, jitter, parallel_start_end, sep_measure):
    '''
    Generates data in increasing number of lines with start_dim dimensions and transforms it to increasing dimensions.

    Returns:
    Heatmap quality of reconstruction and transformation in number of lines vs. transformation dimensions
    '''
    
    pickled = find_in_pickle_lines_vs_dyn_dim("./pickled_results/transf_dyn_red_dyns_vs_neighbours.pickle", [seed, start_dim, k_neigh, order, num_lines, num_points_per_line, bounds, dist_measure, jitter, parallel_start_end, sep_measure])
    # list = [lines] => lines = [dim`s] 
    # list = [[dims`s], [dim's],...]
    #          line 1    line2 ... 
    list_pca_means = []
    list_vector_pca_means = []
    list_transf_pca_means = []
    list_transf_vector_pca_means = []
    dim_list = general_func.generate_integer_list_dyn_low(start_dim, num_lines, order)
    separated = False

    if pickled is not None:
        list_pca_means = pickled[0]
        list_vector_pca_means = pickled[1]
        list_transf_pca_means = pickled[2]
        list_transf_vector_pca_means = pickled[3]

    elif pickled is None:
        # Generate data in current dimensionality
        sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dim, jitter, bounds, parallel_start_end)

        if separated:
            data = flatten(sep_points)
        else:
            data = sep_points

        for k in range(1, k_neigh + 1):
            line_dim_result = []
            vector_line_dim_result = []

            transf_line_dim_result = []
            transf_vector_line_dim_result = []

            for target_dim in dim_list:

                # Transform  and reconstruct data
                reconstr_data, transf_data = pca_reconstruction(data, target_dim, complex_PCA.ComplexPCA, None, None)

                # Reconstruct vector data
                vectors = get_neighbor_vector(sep_points, separated)
                reconstr_comp_data, transf_comp_data = vector_PCA.vector_pca_reconstruction(data, vectors, target_dim)

                # Compute exp-values and variances for reconstructed and transformed data
                if not sep_measure:
                    reconstr_exp_values, _, _ = get_mu_var_by_neighbor_num(data, reconstr_data, dist_measure, k)
                    reconstr_exp_comp_values, _, _ = get_mu_var_by_neighbor_num(data, reconstr_comp_data, dist_measure, k)

                    transf_exp_values, _, _ = get_mu_var_by_neighbor_num(data, transf_data, dist_measure, k)
                    transf_exp_comp_values, _, _ = get_mu_var_by_neighbor_num(data, transf_comp_data, dist_measure, k)

                elif sep_measure:
                    sep_rec_data = [reconstr_data[i:i + num_points_per_line] for i in range(0, len(reconstr_data), num_points_per_line)]
                    sep_rec_comp_data = [reconstr_comp_data[i:i + num_points_per_line] for i in range(0, len(reconstr_comp_data), num_points_per_line)]

                    reconstr_exp_values, _, _ = measure_sep_structures(sep_points, sep_rec_data, dist_measure, k)
                    reconstr_exp_comp_values, _, _ = measure_sep_structures(sep_points, sep_rec_comp_data, dist_measure, k)

                    sep_transf_data = [transf_data[i:i + num_points_per_line] for i in range(0, len(transf_data), num_points_per_line)]
                    sep_transf_comp_data = [transf_comp_data[i:i + num_points_per_line] for i in range(0, len(transf_comp_data), num_points_per_line)]

                    transf_exp_values, _, _ = measure_sep_structures(sep_points, sep_transf_data, dist_measure, k)
                    transf_exp_comp_values, _, _ = measure_sep_structures(sep_points, sep_transf_comp_data, dist_measure, k)

                line_dim_result.append(reconstr_exp_values[-1])
                vector_line_dim_result.append(reconstr_exp_comp_values[-1])

                transf_line_dim_result.append(transf_exp_values[-1])
                transf_vector_line_dim_result.append(transf_exp_comp_values[-1])
            
            list_pca_means.append(line_dim_result)
            list_vector_pca_means.append(vector_line_dim_result)
            list_transf_pca_means.append(transf_line_dim_result)
            list_transf_vector_pca_means.append(transf_vector_line_dim_result)

        pickle_results_lines_vs_dyn_dim("./pickled_results/dyn_red_dyns_vs_neighbours.pickle", [seed, start_dim, k_neigh, order, num_lines, num_points_per_line, bounds, dist_measure, jitter, parallel_start_end, sep_measure], list_pca_means, list_vector_pca_means, list_transf_pca_means, list_transf_vector_pca_means)
        
    fig, axes = plt.subplots(2,2)

    rec_pca = pd.DataFrame(list_pca_means, columns=dim_list, index = list(range(1, k_neigh + 1)))
    sns.heatmap(rec_pca, ax = axes[0, 0], annot=False)
    axes[0, 0].set_title("PCA")
    axes[0, 0].set(xlabel="Transformation dimensionality", ylabel="Number of neighbours")

    rec_vector_pca = pd.DataFrame(list_vector_pca_means, columns=dim_list, index = list(range(1, k_neigh + 1)))
    sns.heatmap(data=rec_vector_pca, ax = axes[0, 1])    
    axes[0, 1].set_title("Vector PCA")
    axes[0, 1].set(xlabel="Transformation dimensionality", ylabel="Number of neighbours")

    transf_pca = pd.DataFrame(list_transf_pca_means, columns=dim_list, index = list(range(1, k_neigh + 1)))
    sns.heatmap(transf_pca, ax = axes[1, 0])   
    axes[1, 0].set_title("Transformed: PCA")
    axes[1, 0].set(xlabel="Transformation dimensionality", ylabel="Number of neighbours")

    transf_vector_pca = pd.DataFrame(list_transf_vector_pca_means, columns=dim_list, index = list(range(1, k_neigh + 1)))
    sns.heatmap(transf_vector_pca, ax = axes[1, 1])   
    axes[1, 1].set_title("Transformed: Vector PCA")
    axes[1, 1].set(xlabel="Transformation dimensionality", ylabel="Number of neighbours")

    fig.suptitle("Original space of dimensionality " + str(start_dim) + " in Order \"" + str(order) + "\" with jitter " + str(jitter) + "\nNumber of points per line: " + str(num_points_per_line) + " and number of lines "+ str(num_lines))
    plt.tight_layout()
    plt.show()


def heatmap_all_techniques_dyn_red_dyns_vs_neighbours(seed, start_dim, k, order, num_lines, num_points_per_line, bounds, dist_measure, jitter, parallel_start_end, sep_measure):
    '''
    Generates data in increasing number of lines with start_dim dimensions and transforms it to increasing dimensions.

    Returns:
    Heatmap quality of reconstruction and transformation in number of lines vs. transformation dimensions
    '''
    sep_points = None
    pickled = find_in_pickle_lines_vs_dyn_dim("./pickled_results/dyn_red_dyns_vs_neighbours.pickle", [seed, start_dim, k, order, num_lines, num_points_per_line, bounds, dist_measure, jitter, parallel_start_end, sep_measure])

    pickled_other_methods = find_in_pickle_lines_vs_dyn_dim("./pickled_results/dyn_red_dyns_vs_neighbours_other_methods.pickle", [seed, start_dim, k, order, num_lines, num_points_per_line, bounds, dist_measure, jitter, parallel_start_end, sep_measure], other_methods = True)

    list_pca_means = []
    list_vector_pca_means = []
    list_transf_pca_means = []
    list_transf_vector_pca_means = []

    list_lle_means = []
    list_tsne_means = []
    list_umap_means = []

    dim_list = general_func.generate_integer_list_dyn_low(start_dim, num_lines, order)

    if pickled is not None:
        list_pca_means = pickled[0]
        list_vector_pca_means = pickled[1]
        list_transf_pca_means = pickled[2]
        list_transf_vector_pca_means = pickled[3]
    else:
        # Generate data in current dimensionality
        if sep_points is None:
            sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dim, jitter, bounds, parallel_start_end, False)
            
            if separated:
                data = flatten(sep_points)
            else:
                data = sep_points
                
            # Compute Vectordata
            vectors = get_neighbor_vector(sep_points, separated)
        
        for target_dim in dim_list:
            # Transform  and reconstruct data
            reconstr_data, transf_data = pca_reconstruction(data, target_dim, complex_PCA.ComplexPCA, None, None)

            # Reconstruct vector data
            reconstr_comp_data, transf_comp_data = vector_PCA.vector_pca_reconstruction(data, vectors, target_dim)

            # Compute exp-values and variances for reconstructed and transformed data
            if not sep_measure:
                reconstr_exp_values, _, _ = get_mu_var_by_neighbor_num(data, reconstr_data, dist_measure, k)
                reconstr_exp_comp_values, _, _ = get_mu_var_by_neighbor_num(data, reconstr_comp_data, dist_measure, k)

                transf_exp_values, _, _ = get_mu_var_by_neighbor_num(data, transf_data, dist_measure, k)
                transf_exp_comp_values, _, _ = get_mu_var_by_neighbor_num(data, transf_comp_data, dist_measure, k)

            elif sep_measure:
                sep_rec_data = [reconstr_data[i:i + num_points_per_line] for i in range(0, len(reconstr_data), num_points_per_line)]
                sep_rec_comp_data = [reconstr_comp_data[i:i + num_points_per_line] for i in range(0, len(reconstr_comp_data), num_points_per_line)]

                reconstr_exp_values, _, _ = measure_sep_structures(sep_points, sep_rec_data, dist_measure, k)
                reconstr_exp_comp_values, _, _ = measure_sep_structures(sep_points, sep_rec_comp_data, dist_measure, k)

                sep_transf_data = [transf_data[i:i + num_points_per_line] for i in range(0, len(transf_data), num_points_per_line)]
                sep_transf_comp_data = [transf_comp_data[i:i + num_points_per_line] for i in range(0, len(transf_comp_data), num_points_per_line)]

                transf_exp_values, _, _ = measure_sep_structures(sep_points, sep_transf_data, dist_measure, k)
                transf_exp_comp_values, _, _ = measure_sep_structures(sep_points, sep_transf_comp_data, dist_measure, k)

            list_pca_means.append(reconstr_exp_values)
            list_vector_pca_means.append(reconstr_exp_comp_values)

            list_transf_pca_means.append(transf_exp_values)
            list_transf_vector_pca_means.append(transf_exp_comp_values)
        
        pickle_results_lines_vs_dyn_dim("./pickled_results/dyn_red_dyns_vs_neighbours.pickle", [seed, start_dim, k, order, num_lines, num_points_per_line, bounds, dist_measure, jitter, parallel_start_end, sep_measure], list_pca_means, list_vector_pca_means, list_transf_pca_means, list_transf_vector_pca_means)

    if pickled_other_methods is not None:
        list_lle_means = pickled_other_methods[0]
        list_tsne_means = pickled_other_methods[1]
        list_umap_means = pickled_other_methods[2]

    else:
        # Generate data in current dimensionality
        if sep_points is None:
            sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dim, jitter, bounds, parallel_start_end, False)
            
            if separated:
                data = flatten(sep_points)
            else:
                data = sep_points
                
            # Compute Vectordata
            vectors = get_neighbor_vector(sep_points, separated)
        
        for target_dim in dim_list:
            # Transform with lle
            lle = LocallyLinearEmbedding(n_components=target_dim, eigen_solver="dense")
            lle_transf_data = lle.fit_transform(data)
            
            # Transform with TSNE
            assert start_dim < num_lines * num_points_per_line
            try:
                tsne = TSNE(n_components=target_dim, method="exact")
            except:
                tsne = TSNE(n_components=target_dim, method="exact", svd_solver="randomized")
            tsne_transf_data = tsne.fit_transform(data)

            # Transform with UMAP
            umap_reducer = UMAP(n_components=target_dim)
            scaled_data = StandardScaler().fit_transform(data)
            umap_transf_data = umap_reducer.fit_transform(scaled_data)

            # Compute exp-values and variances for reconstructed and transformed data
            if not sep_measure:
                transf_lle_exp_values, _, _ = get_mu_var_by_neighbor_num(data, lle_transf_data, dist_measure, k)
                transf_tsne_exp_values, _, _ = get_mu_var_by_neighbor_num(data, tsne_transf_data, dist_measure, k)
                transf_umap_exp_values, _, _ = get_mu_var_by_neighbor_num(data, umap_transf_data, dist_measure, k)

            elif sep_measure:
                sep_lle_transf_data = [lle_transf_data[i:i + num_points_per_line] for i in range(0, len(lle_transf_data), num_points_per_line)]
                sep_tsne_trans_data = [tsne_transf_data[i:i + num_points_per_line] for i in range(0, len(tsne_transf_data), num_points_per_line)]
                sep_umap_trans_data = [umap_transf_data[i:i + num_points_per_line] for i in range(0, len(umap_transf_data), num_points_per_line)]

                transf_lle_exp_values, _, _ = measure_sep_structures(sep_points, sep_lle_transf_data, dist_measure, k)
                transf_tsne_exp_values, _, _ = measure_sep_structures(sep_points, sep_tsne_trans_data, dist_measure, k)
                transf_umap_exp_values, _, _ = measure_sep_structures(sep_points, sep_umap_trans_data, dist_measure, k)

            list_lle_means.append(transf_lle_exp_values)
            list_tsne_means.append(transf_tsne_exp_values)
            list_umap_means.append(transf_umap_exp_values)

        pickle_results_lines_vs_dyn_dim_other_methods("./pickled_results/dyn_red_dyns_vs_neighbours_other_methods.pickle", [seed, start_dim, k, order, num_lines, num_points_per_line, bounds, dist_measure, jitter, parallel_start_end, sep_measure], list_lle_means, list_tsne_means, list_umap_means)
        
    fig, axes = plt.subplots(2,3)
    if k < 10:
        x_tick_labels = list(range(1, k + 1))
        y_tick_labels = dim_list
    elif k < 50:
        x_tick_labels = list(range(1, k + 1, 2))
        y_tick_labels = [d for i, d in enumerate(dim_list) if (i + 1) % 2 == 1]

    sns.heatmap(list_transf_pca_means, ax = axes[0, 0])   
    axes[0, 0].set_title("Transformed: PCA")
    axes[0, 0].set(ylabel="Transformation dimensionality", xlabel="Number of neighbours")
    axes[0, 0].set(xticklabels = x_tick_labels, yticklabels = y_tick_labels)

    sns.heatmap(list_transf_vector_pca_means, ax = axes[0, 1])   
    axes[0, 1].set_title("Transformed: Vector PCA")
    axes[0, 1].set(ylabel="Transformation dimensionality", xlabel="Number of neighbours")
    axes[0, 1].set(xticklabels = x_tick_labels, yticklabels = y_tick_labels)

    sns.heatmap(list_lle_means, ax = axes[1, 0])   
    axes[1, 0].set_title("Transformed:LLE")
    axes[1, 0].set(ylabel="Transformation dimensionality", xlabel="Number of neighbours")
    axes[1, 0].set(xticklabels = x_tick_labels, yticklabels = y_tick_labels)

    sns.heatmap(list_tsne_means, ax = axes[1, 1])   
    axes[1, 1].set_title("Transformed: T-SNE")
    axes[1, 1].set(ylabel="Transformation dimensionality", xlabel="Number of neighbours")
    axes[1, 1].set(xticklabels = x_tick_labels, yticklabels = y_tick_labels)

    sns.heatmap(list_umap_means, ax = axes[1, 2])   
    axes[1, 2].set_title("Transformed: UMAP")
    axes[1, 2].set(ylabel="Transformation dimensionality", xlabel="Number of neighbours")
    axes[1, 2].set(xticklabels = x_tick_labels, yticklabels = y_tick_labels)

    fig.suptitle("Original space of dimensionality " + str(start_dim) + " in Order \"" + str(order) + "\" with jitter " + str(jitter) + "\nNumber of points per line: " + str(num_points_per_line) + " and number of lines "+ str(num_lines))
    plt.tight_layout()
    plt.savefig("./plots/neigh_vs_dim_w_other_methods_order " + str(order) + "_startdim" + str(start_dim) + "_jitter" + str(jitter))

    plt.show()


def show_corr_mats_line_heatmap(start_dimension, target_dimension, order, num_points_per_line, num_lines, bounds, jitter_bound):
  
    sep_points = generate_data(bounds, start_dimension, num_points_per_line, num_lines, jitter_bound)
    sep_points = orderings.order_nearest_start_points(sep_points)
    
    if order == "zigzag":
        data = orderings.zigzag_order(sep_points, num_points_per_line)
    elif order == "one_line":
        data = orderings.one_line(sep_points)
    elif order == "random":
        data = orderings.random_ordering(flatten(sep_points))
    elif order == "connected_zigzag":
        data = generate_connected_zigzag(num_points_per_line, num_lines, start_dimension, bounds, jitter_bound)
    else:
        data = flatten(sep_points)
    
    # Transform data with vector pca
    vector_data = get_neighbor_vector(data, False)
    reconstr_data, transf_data = vector_PCA.vector_pca_reconstruction(flatten(sep_points), vector_data, target_dimension)
    
    correl_coeff_mats, transf_correl_coeff_mats, rec_correl_coeff_mats = measure_correlation_lines(sep_points, transf_data, reconstr_data, num_points_per_line)
    
    fig, axes = plt.subplots(3, len(correl_coeff_mats))
    
    for i, correl_mat in enumerate(correl_coeff_mats):
        sns.heatmap(correl_mat, ax = axes[0, i])   
        axes[0, i].set(xlabel="Points in line " + str(i), ylabel="Points in line " + str(i))
        
        sns.heatmap(transf_correl_coeff_mats[i], ax = axes[1, i])   
        axes[1, i].set(xlabel="Points in line " + str(i), ylabel="Points in line " + str(i))
        
        sns.heatmap(rec_correl_coeff_mats[i], ax = axes[2, i])   
        axes[2, i].set(xlabel="Points in line " + str(i), ylabel="Points in line " + str(i))
    
    plt.tight_layout()
    plt.show()