from sklearn.random_projection import GaussianRandomProjection
import numpy as np
import pickle
import math

def flatten(sep_points):
    return np.array([item for i in sep_points for item in i])


def generate_integer_list_dyn_high(target_dim):
    """
    Generate a list of integers in steps 10 and bigger than k up to 100.

    Parameters:
        k (int): Input integer.

    Returns:
        result_list (list): List of integers.
    """
    result_list = []
    print(target_dim)

    if target_dim < 10:
        result_list.extend(range(target_dim, 10, 1))
        result_list.extend(range(10, 110, 10))
    else:
        target_dim = min(target_dim, 100)
        result_list.extend(range(target_dim, 10, 1))
        result_list.extend(range(10, 110, 10))
    return result_list


def generate_integer_list_dyn_low(start_dim, num_lines, order):
    """
    Generate a list of integers in steps 10 lower than k

    Parameters:
        start_dim (int): Input integer up to which numbers are generated.

    Returns:
        result_list (list): List of integers.
    """
    result_list = []
    if order == "air_pollution":
        result_list.extend(list(range(2, 11)))
        result_list.extend(list(range(20, 300, 25)))
        result_list.extend(list(range(300, start_dim, 1000)))
    elif order == "russell2000_stock":
        result_list.extend(list(range(2, 11)))
        result_list.extend(list(range(20, 250, 25)))
        result_list.extend(list(range(250, start_dim + 100, 200)))
    elif order == "flights":
        result_list.extend(list(range(2, 10)))
        result_list.extend(list(range(12, start_dim, 2)))
    else:
        assert(start_dim >= 1)
        result_list.extend(list(range(2, min(num_lines, start_dim) + 10, math.ceil(min(num_lines, start_dim) / 5))))
        if start_dim > num_lines:
            steps = math.ceil((start_dim / 10)/ 5) * 5
            result_list.extend(range(math.ceil(max(result_list) / 5) * 5, start_dim, steps))
        result_list = list(set(result_list))
        result_list.sort()
    print(result_list)
    return result_list


def label_name(name, i, line=None):
    if i == 0:
        if not line or line == 0:
            return(name)


def random_projection(data, target_dim):
    random_projection = GaussianRandomProjection(n_components=target_dim)
    projected_data = random_projection.fit_transform(data)

    return projected_data, random_projection


def pickle_results(file_path, parameters, exp_values, var_values):
    new_result = {
        "parameters" : parameters,
        "exp_values" : exp_values,
        "var_values" : var_values
    }
    try:
        with open(file_path, 'rb') as file:
            prev_results = pickle.load(file)
    except:
        prev_results = {}
    
    prev_results[len(prev_results) + 1] = new_result
    
    with open(file_path, 'wb') as file:
        pickle.dump(prev_results, file)
        

def find_in_pickle_with_param(file_path, picked_options):
    try:
        with open(file_path, 'rb') as file:
            results = pickle.load(file)
    except:
        print("File does not exist or is empty")
        return None
    
    for i, result in results.items():
        if result["parameters"] == picked_options:
            return result
        
    print("Result not yet pickled")
    return None


def find_in_pickle_all_without_seed(file_path, parameters, compare_last = True):
    try:
        with open(file_path, 'rb') as file:
            results = pickle.load(file)
    except:
        print("File does not exist or is empty")
        return None
    result_list = []
    for i, result in results.items():
        # print(result["parameters"])
        # if result["parameters"][2] == "flights":
        #     print(result["parameters"])
        # seed = result["parameters"].pop(0)
        if not compare_last:
            if result["parameters"][1:-1] == parameters:
                result["parameters"]
                result_list.append(result)
        else:
            if result["parameters"][1:] == parameters:
                # result["parameters"].insert(0, seed)
                result_list.append(result)
    return result_list


def find_in_pickle_for_specific_data(file_path, data_name):
    try:
        with open(file_path, 'rb') as file:
            results = pickle.load(file)
    except:
        print("File does not exist or is empty")
        return None
    
    result_list = []
    for i, result in results.items():
        if result["parameters"][2] == data_name:
            result_list.append(result)
    return result_list

find_in_pickle_for_specific_data("pickled_results/slurm/avg_dev_vs_dyn_low_dims.pickle", "russell2000_stock")


def pickle_results_avg_dev_vs_dyn_dim(file_path, parameters, pca_mean_var, vector_mean_var, rand_proj_means_vars, transf_pca_mean_var, transf_vector_mean_var, transf_random_proj_mean_var):
    new_result = {
        "parameters" : parameters,
        "pca_mean_var" : pca_mean_var,
        "vector_mean_var" : vector_mean_var,
        "rand_proj_means_vars" : rand_proj_means_vars,
        "transf_pca_mean_var" : transf_pca_mean_var,
        "transf_vector_mean_var" : transf_vector_mean_var,
        "transf_random_proj_mean_var" : transf_random_proj_mean_var
    }
    try:
        with open(file_path, 'rb') as file:
            prev_results = pickle.load(file)
    except:
        prev_results = {}
    
    prev_results[len(prev_results) + 1] = new_result
    
    with open(file_path, 'wb') as file:
        pickle.dump(prev_results, file)
        
def pickle_cpca_results_avg_dev_vs_dyn_dim(file_path, parameters, results):
    new_result = {
        "parameters" : parameters,
        "results": results
    }
    try:
        with open(file_path, 'rb') as file:
            prev_results = pickle.load(file)
    except:
        prev_results = {}
    
    prev_results[len(prev_results) + 1] = new_result
    
    with open(file_path, 'wb') as file:
        pickle.dump(prev_results, file)
     
        
def pickle_ica_results_avg_dev_vs_dyn_dim(file_path, parameters, ica_mean_var, vector_ica_mean_var, transf_ica_mean_var, transf_vector_ica_mean_var, transf_tica_means_vars):
    new_result = {
        "parameters" : parameters,
        "ica_mean_var" : ica_mean_var,
        "vector_ica_mean_var" : vector_ica_mean_var,
        "transf_ica_mean_var" : transf_ica_mean_var,
        "transf_vector_ica_mean_var" : transf_vector_ica_mean_var,
        "transf_tica_mean_var" : transf_tica_means_vars,
    }
    try:
        with open(file_path, 'rb') as file:
            prev_results = pickle.load(file)
    except:
        prev_results = {}
    
    prev_results[len(prev_results) + 1] = new_result
    
    with open(file_path, 'wb') as file:
        print("logging")
        pickle.dump(prev_results, file)
        
        
def pickle_rewritten_pickle(file_path, order):
    entry = find_in_pickle_for_specific_data("pickled_results/slurm/avg_dev_vs_dyn_low_dims.pickle", order)
    new_entry = entry[1]
    new_entry["parameters"][3] = [-10000, 65535]
    print(new_entry["parameters"])
    
    pickle_results_avg_dev_vs_dyn_dim("pickled_results/slurm/avg_dev_vs_dyn_low_dims.pickle", new_entry["parameters"], new_entry["pca_mean_var"], new_entry["vector_mean_var"], new_entry["rand_proj_means_vars"], new_entry["transf_pca_mean_var"], new_entry["transf_vector_mean_var"], new_entry["transf_random_proj_mean_var"])
    
# pickle_rewritten_pickle("", "air_pollution")

def find_in_pickle_dyn_dims(file_path, parameters):
    try:
        with open(file_path, 'rb') as file:
            results = pickle.load(file)
    except:
        print("File does not exist or is empty")
        return None
    for i, result in results.items():
        if result["parameters"] == parameters:
            print("Found Run")
            return result["pca_mean_var"], result["vector_mean_var"], result["rand_proj_means_vars"], result["transf_pca_mean_var"], result["transf_vector_mean_var"],  result["transf_random_proj_mean_var"]
        
    print("Result not yet pickled")
    return None


def find_in_pickle_dyn_dims_wo_target_dim(file_path, parameters_wo_target_dim):
    try:
        with open(file_path, 'rb') as file:
            results = pickle.load(file)
    except:
        print("File does not exist or is empty")
        return None
    result_list = []
    print(parameters_wo_target_dim)
    for i, result in results.items():
        if result["parameters"][:-1] == parameters_wo_target_dim:
            print("Found one for target dim ", result["parameters"][-1])
            result_list.append(result)
            # return result["pca_mean_var"], result["vector_mean_var"], result["rand_proj_means_vars"], result["transf_pca_mean_var"], result["transf_vector_mean_var"],  result["transf_random_proj_mean_var"]
        
    return result_list


def find_in_ica_pickle_dyn_dims(file_path, parameters):
    try:
        with open(file_path, 'rb') as file:
            results = pickle.load(file)
    except:
        print("File does not exist or is empty")
        return None
    
    for i, result in results.items():
        if result["parameters"] == parameters:
            print("Found Run")
            return result["ica_mean_var"], result["vector_ica_mean_var"], result["transf_ica_mean_var"], result["transf_vector_ica_mean_var"],  result["transf_tica_mean_var"]
        
    print("Result not yet pickled")
    return None


def find_in_pickle_lines_vs_dyn_dim(file_path, parameters, other_methods = False):
    try:
        with open(file_path, 'rb') as file:
            results = pickle.load(file)
    except:
        print("File does not exist or is empty")
        return None
    
    if other_methods:
        for i, result in results.items():
            if result["parameters"] == parameters:
                print("Found Run")
                return (result["list_lle_means"], result["list_tsne_means"], result["list_umap_means"])
    else:
        for i, result in results.items():
            if result["parameters"] == parameters:
                print("Found Run")
                return (result["list_pca_means"], result["list_vector_pca_means"], result["list_transf_pca_means"], result["list_transf_vector_pca_means"])
        
    print("Result not yet pickled")
    return None


def pickle_results_lines_vs_dyn_dim(file_path, parameters, list_pca_means, list_vector_pca_means, list_transf_pca_means, list_transf_vector_pca_means):
    new_result = {
        "parameters" : parameters,
        "list_pca_means" : list_pca_means,
        "list_vector_pca_means" : list_vector_pca_means,
        "list_transf_pca_means" : list_transf_pca_means,
        "list_transf_vector_pca_means" : list_transf_vector_pca_means
    }
    try:
        with open(file_path, 'rb') as file:
            prev_results = pickle.load(file)
    except:
        prev_results = {}
    
    prev_results[len(prev_results) + 1] = new_result
    
    with open(file_path, 'wb') as file:
        pickle.dump(prev_results, file)
        

def pickle_results_lines_vs_dyn_dim_other_methods(file_path, parameters, list_lle_means, list_tsne_means, list_umap_means):
    new_result = {
        "parameters" : parameters,
        "list_lle_means" : list_lle_means,
        "list_tsne_means" : list_tsne_means,
        "list_umap_means" : list_umap_means,
    }
    try:
        with open(file_path, 'rb') as file:
            prev_results = pickle.load(file)
    except:
        prev_results = {}
    
    prev_results[len(prev_results) + 1] = new_result
    
    with open(file_path, 'wb') as file:
        pickle.dump(prev_results, file)
        

def pickle_results_scalar_product(file_path, parameters, rec_pca_scal_prods, transf_pca_scal_prods, rec_vector_pca_scal_prods, transf_vector_pca_scal_prods):
    new_result = {
        "parameters": parameters,
        "rec_pca_scal_prods": rec_pca_scal_prods,
        "transf_pca_scal_prods": transf_pca_scal_prods,
        "rec_vector_pca_scal_prods": rec_vector_pca_scal_prods,
        "transf_vector_pca_scal_prods": transf_vector_pca_scal_prods
    }

    try:
        with open(file_path, 'rb') as file:
            prev_results = pickle.load(file)
    except:
        prev_results = {}
    
    prev_results[len(prev_results) + 1] = new_result
    
    with open(file_path, 'wb') as file:
        pickle.dump(prev_results, file)
        

def pickle_neighbour_error(file_path, parameters, pca_results, vector_pca_results, rand_proj_results, transf_pca_results, transf_vector_pca_results, transf_rand_proj_results):
    new_result = {
        "parameters": parameters,
        "pca_results": pca_results,
        "vector_pca_results": vector_pca_results,
        "rand_proj_results": rand_proj_results,
        "transf_pca_results": transf_pca_results,
        "transf_vector_pca_results": transf_vector_pca_results,
        "transf_rand_proj_results": transf_rand_proj_results
    }

    try:
        with open(file_path, 'rb') as file:
            prev_results = pickle.load(file)
    except:
        prev_results = {}
    
    prev_results[len(prev_results) + 1] = new_result
    
    with open(file_path, 'wb') as file:
        pickle.dump(prev_results, file)
        

def pickle_neighb_error_other_techniques(file_path, parameters, lle_results, tsne_results, umap_results):
    new_result = {
        "parameters": parameters,
        "lle_results": lle_results,
        "tsne_results": tsne_results,
        "umap_results": umap_results
    }

    try:
        with open(file_path, 'rb') as file:
            prev_results = pickle.load(file)
    except:
        prev_results = {}
    
    prev_results[len(prev_results) + 1] = new_result
    
    with open(file_path, 'wb') as file:
        pickle.dump(prev_results, file)


def pickle_num_neigh_vs_dyn_low_dim(file_path, parameters, list_pca_means, list_vector_means, list_rand_proj_means, list_transf_pca_means, list_transf_vector_means, list_transf_rand_proj_means, list_lle_means, list_tsne_means, list_umap_means):
    
    new_pickle = {
        "parameters" : parameters,
        "list_pca_means" : list_pca_means,
        "list_vector_means" : list_vector_means,
        "list_rand_proj_means" : list_rand_proj_means,
        "list_transf_pca_means" : list_transf_pca_means, 
        "list_transf_vector_means" : list_transf_vector_means, 
        "list_transf_rand_proj_means" : list_transf_rand_proj_means,
        "list_lle_means" : list_lle_means,
        "list_tsne_means" : list_tsne_means,
        "list_umap_means" : list_umap_means
    }
    try:
        with open(file_path, 'rb') as file:
            prev_results = pickle.load(file)
    except:
        prev_results = {}
    
    prev_results[len(prev_results) + 1] = new_pickle
    
    with open(file_path, 'wb') as file:
        pickle.dump(prev_results, file)