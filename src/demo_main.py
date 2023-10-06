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
from plot_func.demo_plots import *

def main():
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    picked_options = get_picked_options()
    
    # plot_s_vector_pca_transformation(1000)

    # plot_s_umap_transformation(1000)
    plot_s_pca_transformation(1000)
    plot_s_lle_transformation(1000)
    plot_s_data(1000)
    
    # plot_lines_data("zigzag", 100, 2, [0, 100])
    # plot_lines_pca_transformation("zigzag", 100, 2, [0, 100], 1)
    # plot_lines_vector_pca_transformation("zigzag", 100, 2, [0, 100], 1)
    
    # plot_demo_pca_arrows(26, "zigzag")
    # plot_demo_pca_arrows(26, "one_line")
    
    
main()