from sklearn.manifold import LocallyLinearEmbedding, TSNE
from structure_measures import *
from multiple_runs import multiple_runs_scalar_product
import matplotlib.pyplot as plt
from data_generation import *
from general_func import *
from complex_PCA import *
from vector_PCA import *
import seaborn as sns
import general_func
import complex_PCA
import numpy as np
import matplotlib
import orderings
import real_PCA
import matplotlib
matplotlib.rcParams.update({'font.size': 26})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["axes.formatter.useoffset"] = False
matplotlib.rcParams['axes.titlepad'] = 20


def show_cov(sep_points, dimensions, num_lines, num_points):
    # Get data ordered by line and zigzag with neighborvectors separately and as whole complex data
    one_line = orderings.one_line(sep_points)
    one_line_vectors = get_neighbor_vector(one_line, False)
    complex_lines = add_complex_vector(one_line, False)

    one_line_meaned = one_line - np.mean(one_line, axis=0, keepdims=True)
    one_line_vectors_meaned = one_line_vectors - np.mean(one_line_vectors, axis=0, keepdims=True)
    complex_lines_meaned = complex_lines - np.mean(complex_lines, axis=0, keepdims=True)
    
    zigzag = orderings.zigzag_order(sep_points, num_points)
    zigzag_vectors = get_neighbor_vector(zigzag, False)
    complex_zigzag = add_complex_vector(zigzag, False)
    
    zigzag_meaned = zigzag - np.mean(zigzag, axis=0, keepdims=True)
    zigzag_vectors_meaned = zigzag_vectors - np.mean(zigzag_vectors, axis=0, keepdims=True)
    complex_zigzag_meaned = complex_zigzag - np.mean(complex_zigzag)

    # Calculate cov
    cov_one_line = np.cov(one_line_meaned, bias=True)
    cov_line_vectors = np.cov(one_line_vectors_meaned, bias=True)
    cov_complex_lines = np.cov(complex_lines_meaned, bias=True)
    cov_zigzag = np.cov(zigzag_meaned, bias=True)
    cov_zigzag_vectors = np.cov(zigzag_vectors_meaned, bias=True)
    cov_complex_zigzag = np.cov(complex_zigzag_meaned, bias=True)

    fig, ax = plt.subplots(2,2)

    ax = plt.subplot(2,2,1)
    sns.heatmap(cov_one_line, ax=ax)
    ax.set_title("Lines: normal")

    ax = plt.subplot(2,2,2)
    sns.heatmap(cov_line_vectors, ax=ax)
    ax.set_title("Lines: vectors")

    ax = plt.subplot(2,2,3)
    sns.heatmap(cov_complex_lines.real, ax=ax)
    ax.set_title("Complex lines: real")

    ax = plt.subplot(2,2, 4)
    sns.heatmap(cov_complex_lines.imag, ax=ax)
    ax.set_title("Complex lines: imaginary")

    fig, ax = plt.subplots(2,2)

    ax = plt.subplot(2,2,1)
    sns.heatmap(cov_zigzag, ax=ax)
    ax.set_title("Zigzag: normal")

    ax = plt.subplot(2,2,2)
    sns.heatmap(cov_zigzag_vectors, ax=ax)
    ax.set_title("Zigzag: vectors")

    ax = plt.subplot(2,2,3)
    sns.heatmap(cov_complex_zigzag.real, ax=ax)
    ax.set_title("Complex zigzag: real")

    ax = plt.subplot(2,2,4)
    sns.heatmap(cov_complex_zigzag.imag, ax=ax)
    ax.set_title("Complex zigzag: imaginary")

    plt.show()


def show_rev_cov(sep_points, dimensions, num_lines, num_points):
    zigzag = orderings.zigzag_order(sep_points, num_points)
    one_line = orderings.one_line(sep_points)

    complex_one_line = add_complex_vector(one_line, False)
    complex_one_line_meaned = complex_one_line - np.mean(complex_one_line,axis=0,keepdims=True)
    cov_complex_lines = np.cov(complex_one_line_meaned, bias=True)

    rev_complex_one_line = orderings.reverse_ordering(complex_one_line)
    rev_complex_one_line_mean = rev_complex_one_line - np.mean(rev_complex_one_line,axis=0,keepdims=True)
    cov_rev_complex_lines = np.cov(rev_complex_one_line_mean, bias=True)

    complex_zigzag = add_complex_vector(zigzag, False)
    complex_zigzag_mean = complex_zigzag - np.mean(complex_zigzag,axis=0,keepdims=True)
    cov_complex_zigzag = np.cov(complex_zigzag_mean, bias=True)

    rev_complex_zigzag = orderings.reverse_ordering(complex_zigzag)
    rev_complex_zigzag_mean = rev_complex_zigzag - np.mean(rev_complex_zigzag,axis=0,keepdims=True)
    cov_rev_complex_zigzag = np.cov(rev_complex_zigzag_mean, bias=True)

    fig, ax = plt.subplots(2,2)

    ax = plt.subplot(2,2,1)
    sns.heatmap(cov_complex_lines.real, ax=ax)
    ax.set_title("Whole complex lines: real")

    ax = plt.subplot(2,2,2)
    sns.heatmap(cov_complex_lines.imag, ax=ax)
    ax.set_title("Whole complex lines: imaginary")

    ax = plt.subplot(2,2,3)
    sns.heatmap(cov_rev_complex_lines.real, ax=ax)
    ax.set_title("Rev whole complex lines: real")

    ax = plt.subplot(2,2,4)
    sns.heatmap(cov_rev_complex_lines.imag, ax=ax)
    ax.set_title("Rev whole complex lines: imaginary")

    fig, ax = plt.subplots(2,2)

    ax = plt.subplot(2,2,1)
    sns.heatmap(cov_complex_zigzag.real, ax=ax)
    ax.set_title("Whole complex zigzag: real")

    ax = plt.subplot(2,2,2)
    sns.heatmap(cov_complex_zigzag.imag, ax=ax)
    ax.set_title("Whole complex zigzag: imaginary")

    ax = plt.subplot(2,2,3)
    sns.heatmap(cov_rev_complex_zigzag.real, ax=ax)
    ax.set_title("Whole rev complex zigzag: real")

    ax = plt.subplot(2,2,4)
    sns.heatmap(cov_rev_complex_zigzag.imag, ax=ax)
    ax.set_title("Whole rev complex zigzag: imaginary")

    plt.show()


# Get line_data -> order data by wished ordering -> create complex data -> PCA -> plot
def plot_complex_points(sep_points, num_points, num_lines, dimensions, order):
    fig, ax = plt.subplots()
    dim = str(dimensions) + 'd'

    try:
        ax = fig.add_subplot(projection=dim)
    except:
        pass

    flat_data = flatten(sep_points)

    # Orderings and neighbor vectors
    next_vectors = []
    data = []
    if order == "sep_lines":
        data = flat_data
        reverse_lines = orderings.reverse_ordering(flat_data)
        next_vectors = get_neighbor_vector(sep_points, True)
    elif order == "zigzag":
        data = orderings.zigzag_order(sep_points)
        reverse_zigzag = orderings.reverse_ordering(data)
        next_vectors = get_neighbor_vector(data, False)
    elif order == "one_line":
        data = orderings.one_line(sep_points)
        reverse_one_line = orderings.reverse_ordering(data)
        next_vectors = get_neighbor_vector(data, False)
    
    # Make complex
    complex_data = add_complex_vector(data, False)
    complex_data_reverse = orderings.reverse_ordering(complex_data)

    # Plot data
    scatter = ax.plot(*zip(*data), 'o', label="datapoints", color= "#FF0000")
    
    ## Add annotations for separate real lines
    annot_list = []
    if order == "sep_lines":
        for i, line in enumerate(sep_points):
            for j, point in enumerate(line):
                annotation = ax.text(*point, str(i) + "." + str(j))
                annot_list.append(annotation)
    else:
        for i, point in enumerate(data):
                annotation = ax.text(*point, str(i))
                annot_list.append(annotation)

    # Separate imaginary points
    # The imaginary parts of complex data and reversed complex data are the same
    imag_line = [[i.imag for i in p] for p in complex_data]
    imag_scatter = ax.plot(*zip(*imag_line),'*', label="vectors", color = "#FF0000")
    
    # PCA on normal data
    pca_normal_data = []
    mean, v_list = complex_PCA.complex_pca(data, dimensions)
    for i in range(0, len(v_list)):
        pca_normal_arrow = ax.plot(*zip(*[mean, (mean + v_list[i])]), label=label_name("PCA on normal data", i), marker=">", markevery=[-1,-1], color="#3333FF")
        pca_normal_data.append(pca_normal_arrow)
    
    # PCA on vector data
    # Again: the vectors are the same for the original and the reversed data
    pca_vector_data = []
    mean, v_list = complex_PCA.complex_pca(next_vectors, dimensions)
    for i in range(0, len(v_list)):
        pca_vector_arrow = ax.plot(*zip(*[mean, (mean + v_list[i])]), label=label_name("PCA on vectors", i), marker="<", markevery=[-1,-1], color="#3333FF", linestyle="dashed")
        pca_vector_data.append(pca_vector_arrow)

    # Complex PCA on whole complex data
    pca_arrow_real = []
    pca_arrow_imag = []
    mean, v_list = complex_PCA.complex_pca(complex_data, dimensions)
    real_mean = [j.real for j in mean]
    imag_mean = [j.imag for j in mean]
    for i in range(0, len(v_list)):
        real_vec = [p.real for p in mean + v_list[i]]
        imag_vec = [p.imag for p in mean + v_list[i]]
        pca_real_arrow = ax.plot(*zip(*[real_mean, real_vec]), label=label_name("Complex PCA: real", i), color = "#FF8000", marker=">", markevery=[-1,-1])
        pca_imag_arrow = ax.plot(*zip(*[imag_mean, imag_vec]), label=label_name("Complex PCA: imaginary", i), color = "#FF8000", marker=">", markevery=[-1,-1], linestyle="dashed")

        pca_arrow_real.append(pca_real_arrow)
        pca_arrow_imag.append(pca_imag_arrow)

    # Complex PCA on whole reversed complex data
    pca_rev_arrow_real = []
    pca_rev_arrow_imag = []
    mean, v_list = complex_PCA.complex_pca(complex_data_reverse, dimensions)
    real_mean = [j.real for j in mean]
    imag_mean = [j.imag for j in mean]
    for i in range(0, len(v_list)):
        real_vec = [p.real for p in mean + v_list[i]]
        imag_vec = [p.imag for p in mean + v_list[i]]
        pca_real_arrow = ax.plot(*zip(*[real_mean, real_vec]), label=label_name("Reversed complex PCA: real", i), color = "#CC0000", marker=">", markevery=[-1,-1])
        pca_imag_arrow = ax.plot(*zip(*[imag_mean, imag_vec]), label=label_name("Reversed complex PCA: imaginary", i), color = "#CC0000", marker=">", markevery=[-1,-1], linestyle="dashed")

        pca_rev_arrow_real.append(pca_real_arrow)
        pca_rev_arrow_imag.append(pca_imag_arrow)
    
    # Create toggle legend
    legend = ax.legend(loc='upper left')
    legends = legend.get_lines()

    for leg in legends:
        leg.set_picker(True)
        leg.set_pickradius(10)

    real_scatter_legend, imag_scatter_legend, pca_normal_legend, pca_vector_legend, whole_pca_real_legend, whole_pca_imag_legend, whole_pca_rev_real_legend, whole_pca_rev_imag_legend = legend.get_lines()

    graphs = {}
    graphs[real_scatter_legend] = [scatter, annot_list]
    graphs[imag_scatter_legend] = [imag_scatter]
    graphs[whole_pca_real_legend] = pca_arrow_real
    graphs[whole_pca_imag_legend] = pca_arrow_imag
    graphs[pca_normal_legend] = pca_normal_data
    graphs[pca_vector_legend] = pca_vector_data
    graphs[whole_pca_rev_real_legend] = pca_rev_arrow_real
    graphs[whole_pca_rev_imag_legend] = pca_rev_arrow_imag

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        try:
            graphs[legend].set_visible(not isVisible)
        except:
            for i in graphs[legend]:
                try:
                    for j in i:
                        j.set_visible(not isVisible)
                except:
                    i[0].set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()

    plt.connect('pick_event', on_pick)
    plt.title(order)
    ax.set_aspect('equal')
    plt.show()


def plot_points(sep_points, part_points):
    num_points = len(sep_points[0])
    dimensions = len(sep_points[0][0])

    fig, ax = plt.subplots()
    dim = str(dimensions) + 'd'

    zigzag = orderings.zigzag_order(sep_points, num_points)
    oneline = orderings.one_line(sep_points)

    try:
        ax = fig.add_subplot(projection=dim)
    except:
        pass

    # PCA for all data in one line
    pca_line_arrow_list = []
    mean, v_list = real_PCA.pca(oneline, dimensions)
    for i in range(0, len(v_list)):
        pca_arrow = ax.plot(*zip(*[mean, (mean + v_list[i])]), label=label_name("pca_line", i), color = "k", marker=">", markevery=[-1,-1])
        pca_line_arrow_list.append(pca_arrow)

    # PCA for zigzag ordered data
    pca_zigzag_arrow_list = []
    mean, v_list = real_PCA.pca(zigzag, dimensions)
    for i in range(0, len(v_list)):
        pca_arrow = ax.plot(*zip(*[mean, (mean+ v_list[i])]), label=label_name("pca_zigzag", i), color = "k", marker=">", markevery=[-1,-1])
        pca_zigzag_arrow_list.append(pca_arrow)

    # Plot lines
    lin_lines = []
    for i, line in enumerate(sep_points):
        line_plot = ax.plot(*zip(*line), label=label_name("line", i))
        lin_lines.append(line_plot)
        
    # Plot all lines connected
    one_line_plot, = ax.plot(*zip(*oneline), label="one line")

    # Plot zigzag line
    zigzag, = ax.plot(*zip(*zigzag), label='zigzag')

    # Plot all data points
    for i in sep_points:
        ax.scatter(*zip(*i))

    # Show PCA for parts of data
    part_list = []
    for j, line in enumerate(part_points):
        mean, v_list = real_PCA.pca(line, dimensions)
        for i in range(0, len(v_list)):
            pca_arrow = ax.plot(*zip(*[mean, (mean + v_list[i])]), label=label_name("pca partition", i, j), color = "0.8", marker=">", markevery=[1,-1])
            part_list.append(pca_arrow)

    # Create toggle legend
    legend = ax.legend(loc='upper right')
    pca_line_legend, pca_zigzag_legend, line_legend, one_line_legend, zigzag_legend, pca_part_legend = legend.get_lines()

    line_legend.set_picker(True)
    line_legend.set_pickradius(10)
    one_line_legend.set_picker(True)
    one_line_legend.set_pickradius(10)
    zigzag_legend.set_picker(True)
    zigzag_legend.set_pickradius(10)
    pca_line_legend.set_picker(True)
    pca_line_legend.set_pickradius(10)
    pca_zigzag_legend.set_picker(True)
    pca_zigzag_legend.set_pickradius(10)
    pca_part_legend.set_picker(True)
    pca_part_legend.set_pickradius(10)

    graphs = {}
    graphs[line_legend] = lin_lines
    graphs[one_line_legend] = one_line_plot
    graphs[zigzag_legend] = zigzag
    graphs[pca_line_legend] = pca_line_arrow_list
    graphs[pca_zigzag_legend] = pca_zigzag_arrow_list
    graphs[pca_part_legend] = part_list

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        try:
            graphs[legend].set_visible(not isVisible)
        except:
            for i in graphs[legend]:
                i[0].set_visible(not isVisible)
        legend.set_visible(not isVisible)

        fig.canvas.draw()
    ax.set_aspect('equal')

    plt.connect('pick_event', on_pick)
    plt.show()


def plot_pca_normal_and_reconstr(picked_options):
    '''Plot original data and reconstructed data'''

    # Get the correct data structure
    sep_points = picked_options["sep_points"]
    separated = picked_options["separated"]

    complex_data = add_complex_vector(sep_points, separated)
    vec_data = get_neighbor_vector(sep_points, separated)

    if separated:
        data = flatten(sep_points)
    else:
        data = sep_points
    
    rec_data, _ = pca_reconstruction(data, picked_options["target_dimension"], complex_PCA.ComplexPCA, None, None)

    rec_vec_data, _ = pca_reconstruction(vec_data, picked_options["target_dimension"], picked_options["pca_choice"], None, None)
    
    rec_comp_data, _ = pca_reconstruction(complex_data, picked_options["target_dimension"], complex_PCA.ComplexPCA_svd_complex, picked_options["scaling_choice"],None)

    fig, ax = plt.subplots()
    assert(picked_options["starting_dimension"] < 4)
    dim = str(picked_options["starting_dimension"]) + 'd'

    try:
        ax = fig.add_subplot(projection=dim)
    except:
        pass

    # Plot normal data points with annotation
    normal_scatter = ax.plot(*zip(*data), 'o', label="original data")
    annot_list = []
    for i, point in enumerate(data):
                annotation = ax.text(*point, str(i))
                annot_list.append(annotation)

    # Plot real part of complex data
    real_complex_scatter = ax.plot(*zip(*complex_data.real), 'o', label="Whole complex data: real")

    # Plot arrows 
    imag_complex_arrows = []
    for i, p in enumerate(complex_data):
        arrow = ax.plot(*zip(*[p.real, p.real + p.imag]), label = label_name("Whole complex data: imaginary", i), color="#2F4F4F", marker=">", markevery=[-1,-1])
        imag_complex_arrows.append(arrow)

    # Plot reconstructed normal data
    reconstr_scatter = ax.plot(*zip(*rec_data), 'o', label="PCA reconstructed original data")

    # Plot reconstructed arrows of normal data
    reconst_vector_arrows = []
    for i, p in enumerate(rec_vec_data):
        # if when_shuffle:
        #     reconstr_normal_transformed = orderings.unshuffle_list(reconstr_normal_transformed, 123456)
        arrow = ax.plot(*zip(*[rec_data[i], rec_data[i] + p]), label = label_name("PCA: original vectors", i), marker=">", markevery=[-1,-1], color="#2F4F4F")
        reconst_vector_arrows.append(arrow)

    # Plot real part of reconstructed complex data
    real_reconst_complex_scatter = ax.plot(*zip(*rec_comp_data.real), 'o', label=picked_options["pca_choice"].whoami() + ": real")
    
    # Plot imaginary part of reconstructed complex data
    imag_reconstr_compl_arrows = []
    for i, p in enumerate(rec_comp_data):
        arrow = ax.plot(*zip(*[p.real, p.real + p.imag]), label = label_name(picked_options["pca_choice"].whoami() + ": imaginary", i), color="#828282", marker=">", markevery=1)
        imag_reconstr_compl_arrows.append(arrow)

    # Add toggle legend
    legend = ax.legend(loc='upper right')
    legends = legend.get_lines()

    for leg in legends:
        leg.set_picker(True)
        leg.set_pickradius(10)

    normal_scatter_legend, real_complex_scatter_legend, imag_complex_arrows_legend, reconstr_scatter_legend, reconst_vectors_legend, real_reconst_complex_scatter_legend, reconstr_arrows_legend = legend.get_lines()
    graphs = {}

    graphs[normal_scatter_legend] = [normal_scatter, annot_list]
    graphs[real_complex_scatter_legend] = [real_complex_scatter]
    graphs[imag_complex_arrows_legend] = imag_complex_arrows
    graphs[reconstr_scatter_legend] = [reconstr_scatter]
    graphs[reconst_vectors_legend] = reconst_vector_arrows
    graphs[real_reconst_complex_scatter_legend] = [real_reconst_complex_scatter]
    graphs[reconstr_arrows_legend] = imag_reconstr_compl_arrows

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        try:
            graphs[legend].set_visible(not isVisible)
        except:
            for i in graphs[legend]:
                try:
                    for j in i:
                        j.set_visible(not isVisible)
                except:
                    i[0].set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()

    plt.connect('pick_event', on_pick)
    plt.title("Original & Reconstructed Data")

    ax.set_aspect('equal')

    plt.show()


def plot_pca_normal_and_reconstr_with_id(picked_options):
    '''Plot original data and reconstructed data'''
    assert(picked_options["starting_dimension"] < 4)
    
    fig, ax = plt.subplots()
    dim = str(picked_options["starting_dimension"]) + 'd'

    try:
        ax = fig.add_subplot(projection=dim)
    except:
        pass

    # Get the correct data structure
    data = picked_options["sep_points"]
    separated = picked_options["separated"]
        
    complex_data = add_complex_vector(data, separated)
    vectors = get_neighbor_vector(data, separated)
    id_data = add_id(data, separated)

    if separated:
        data = general_func.flatten(data)
        id_data = flatten(id_data)

    # Transform and reconstruct data with pca
    reconstr_normal_transformed, _ = pca_reconstruction(data, picked_options["target_dimension"], picked_options["pca_choice"], None, picked_options["when_shuffle"])

    # Transform and reconstruct vectors with pca
    reconstr_vectors_transformed, _ = pca_reconstruction(vectors, picked_options["target_dimension"], picked_options["pca_choice"], None, picked_options["when_shuffle"])

    # Transform and reconstruct data with either complex PCA or vector pca
    if picked_options["scaling_choice"] != "vector_pca":
        reconstr_complex_transformed, _ = pca_reconstruction(complex_data, picked_options["target_dimension"], picked_options["pca_choice"], picked_options["scaling_choice"], picked_options["when_shuffle"])
    else:
        reconstr_complex_transformed, _ = vector_PCA.vector_pca_reconstruction(np.real(complex_data), np.imag(complex_data), picked_options["target_dimension"])

    reconstr_id_transformed, _ = pca_reconstruction(id_data, picked_options["target_dimension"], picked_options["pca_choice"], None, picked_options["when_shuffle"])

    # Plot normal data points with annotation
    normal_scatter = ax.plot(*zip(*data), 'o', label="original data")
    annot_list = []
    for i, point in enumerate(data):
                annotation = ax.text(*point, str(i))
                annot_list.append(annotation)

    # Plot real part of complex data
    real_complex_scatter = ax.plot(*zip(*complex_data.real), 'o', label="Whole complex data: real")

    # Plot arrows 
    imag_complex_arrows = []
    for i, p in enumerate(complex_data):
        arrow = ax.plot(*zip(*[p.real, p.real + p.imag]), label = label_name("Whole complex data: imaginary", i), color="#2F4F4F", marker=">", markevery=[-1,-1])
        imag_complex_arrows.append(arrow)

    # Plot reconstructed normal data
    reconstr_scatter = ax.plot(*zip(*reconstr_normal_transformed), 'o', label="PCA reconstructed original data")

    # Plot reconstructed arrows of normal data
    reconst_vector_arrows = []
    for i, p in enumerate(reconstr_vectors_transformed):
        # if when_shuffle:
        #     reconstr_normal_transformed = orderings.unshuffle_list(reconstr_normal_transformed, 123456)
        arrow = ax.plot(*zip(*[reconstr_normal_transformed[i], reconstr_normal_transformed[i] + p]), label = label_name("PCA: original vectors", i), marker=">", markevery=[-1,-1], color="#2F4F4F")
        reconst_vector_arrows.append(arrow)

    # Plot real part of reconstructed complex data
    if picked_options["scaling_choice"] != "vector_pca":
        real_reconst_complex_scatter = ax.plot(*zip(*reconstr_complex_transformed.real), 'o', label= "complex pca: real")
        
        # Plot imaginary part of reconstructed complex data
        imag_reconstr_compl_arrows = []
        for i, p in enumerate(reconstr_complex_transformed):
            arrow = ax.plot(*zip(*[p.real, p.real + p.imag]), label = label_name("complex pca: imaginary", i), color="#828282", marker=">", markevery=1)
            imag_reconstr_compl_arrows.append(arrow)
    else:
        real_reconst_complex_scatter = ax.plot(*zip(*reconstr_complex_transformed.real), 'o', label= "vector-pca")
        imag_reconstr_compl_arrows = []
        arrow = ax.plot(*([0]*picked_options["starting_dimension"]), label = "ccc", color="#828282", marker=">", markevery=1)
        
    # Plot reconstructed data with id
    reconst_id_scatter = []
    reconst_id_annot = []
    for i, p in enumerate(reconstr_id_transformed):
        point = p[:picked_options["starting_dimension"]]
        enum = p[-1]
        cmap = matplotlib.colormaps["Purples"]
        color_i = (len(reconstr_id_transformed) - i) / len(reconstr_id_transformed)
        rgba = cmap(color_i)
        plot = ax.plot(*point, 'o', label=label_name("ID's", i), color=rgba)
        annot = ax.text(*point, str(round(enum, 2)))
        reconst_id_scatter.append(plot)
        reconst_id_annot.append(annot)

    # Add toggle legend
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    legends = legend.get_lines()
    for leg in legends:
        leg.set_picker(True)
        leg.set_pickradius(10)

    normal_scatter_legend, real_complex_scatter_legend, imag_complex_arrows_legend, reconstr_scatter_legend, reconst_vectors_legend, real_reconst_complex_scatter_legend, reconstr_arrows_legend, id_scatter_legend = legend.get_lines()
    graphs = {}

    graphs[normal_scatter_legend] = [normal_scatter, annot_list]
    graphs[real_complex_scatter_legend] = [real_complex_scatter]
    graphs[imag_complex_arrows_legend] = imag_complex_arrows
    graphs[reconstr_scatter_legend] = [reconstr_scatter]
    graphs[reconst_vectors_legend] = reconst_vector_arrows
    graphs[real_reconst_complex_scatter_legend] = [real_reconst_complex_scatter]
    graphs[reconstr_arrows_legend] = imag_reconstr_compl_arrows
    graphs[id_scatter_legend] = [*reconst_id_scatter, reconst_id_annot]

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        try:
            graphs[legend].set_visible(not isVisible)
        except:
            for i in graphs[legend]:
                try:
                    for j in i:
                        j.set_visible(not isVisible)
                except:
                    i[0].set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()

    plt.connect('pick_event', on_pick)
    plt.title("Original & Reconstructed Data")

    ax.set_aspect('equal')

    plt.show()


def plot_pca_transformed_with_id(picked_options):
    '''Plot original data and reconstructed data'''
    fig, ax = plt.subplots()
    dim = str(picked_options["target_dimension"]) + 'd'

    try:
        ax = fig.add_subplot(projection=dim)
    except:
        pass

    # Get the correct data structure
    data = picked_options["sep_points"]
    separated = picked_options["separated"]
    
    complex_data = add_complex_vector(data, separated)
    vectors = get_neighbor_vector(data, separated)
    id_data = add_id(data, separated)
    if separated:
        data = general_func.flatten(data)

    # Transform data with pca
    _, orig_transformed = pca_transform_data(data, picked_options["target_dimension"], picked_options["pca_choice"], picked_options["scaling_choice"], picked_options["when_shuffle"])

    # Transform vectors with pca
    _, orig_vector_transformed = pca_transform_data(vectors, picked_options["target_dimension"], picked_options["pca_choice"], picked_options["scaling_choice"], picked_options["when_shuffle"])

    # Transform data with either complex pca or vector_pca
    if picked_options["scaling_choice"] != "vector_pca":
        _, complex_transformed = pca_transform_data(complex_data, picked_options["target_dimension"], picked_options["pca_choice"], picked_options["scaling_choice"], picked_options["when_shuffle"])
    else:
        _, complex_transformed = vector_PCA.vector_pca_transform_data(np.real(complex_data), np.imag(complex_data), picked_options["target_dimension"])
    
    # Transform ID data
    _, id_transformed = pca_transform_data(id_data, picked_options["target_dimension"], picked_options["pca_choice"], picked_options["scaling_choice"], picked_options["when_shuffle"])

    # Plot real part of complex data
    orig_scatter = ax.plot(*zip(*orig_transformed), 'o', label="Original data")

    # Plot arrows 
    orig_arrows = []
    for i, p in enumerate(orig_vector_transformed):
        if i < len(orig_vector_transformed) - 1:
            arrow = ax.plot(*zip(*[orig_transformed [i], orig_transformed[i] + p]), label = label_name("Original vectors", i), color="#2F4F4F", marker=">", markevery=[-1,-1])
        orig_arrows.append(arrow)

    # Plot method 1 or 2
    complex_scatter = ax.plot(*zip(*complex_transformed.real), 'o', label="Complex PCA: real")

    # Plot reconstructed arrows of normal data
    complex_vector_arrows = []
    for i, p in enumerate(complex_transformed):
        arrow = ax.plot(*zip(*[p.real, p.real + p.imag]), label = label_name("Complex PCA: imag", i), marker=">", markevery=[-1,-1], color="#2F4F4F")
        complex_vector_arrows.append(arrow)

    # Plot real part of reconstructed complex data
    id_scatter = ax.plot(*zip(*id_transformed.real), 'o', label="ID's")
    
    # Plot imaginary part of reconstructed complex data
    id_arrows = []
    # id_transformed = id_transformed[id_transformed[:, -1].argsort()]
    for i, p in enumerate(id_transformed):
        if i < len(id_transformed) - 1:
            arrow = ax.plot(*zip(*[p, id_transformed[i + 1]]), label = label_name("ID's: vectors", i), color="#828282", marker=">", markevery=1)
            id_arrows.append(arrow)

    # Add toggle legend
    legend = ax.legend(loc='upper right')
    legends = legend.get_lines()
    for leg in legends:
        leg.set_picker(True)
        leg.set_pickradius(10)

    orig_scatter_legend, orig_vector_legend, complex_scatter_legend, complex_arrows_legend, id_scatter_legend , id_arrow_legend= legend.get_lines()
    graphs = {}

    graphs[orig_scatter_legend] = [orig_scatter]
    graphs[orig_vector_legend] = orig_arrows
    graphs[complex_scatter_legend] = [complex_scatter]
    graphs[complex_arrows_legend] = complex_vector_arrows
    graphs[id_scatter_legend] = [id_scatter]
    graphs[id_arrow_legend] = id_arrows

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        try:
            graphs[legend].set_visible(not isVisible)
        except:
            for i in graphs[legend]:
                try:
                    for j in i:
                        j.set_visible(not isVisible)
                except:
                    i[0].set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()

    plt.connect('pick_event', on_pick)
    plt.title("Original & Reconstructed Data")

    ax.set_aspect('equal')

    plt.show()


def plot_transformed_data(picked_options):
    normal_transformed, normal_vector_transformed, complex_transformed, _ = compare_pca_transformed_by_order(picked_options)

    fig, ax = plt.subplots()
    dim = str(picked_options["target_dimension"]) + 'd'
    try:
        ax = fig.add_subplot(projection=dim)
    except:
        pass

    if picked_options["target_dimension"] == 1:
        ax.plot(range(len(normal_transformed)), label=picked_options["order"])
        normal_transformed_scatter = ax.plot(general_func.flatten(normal_transformed), np.zeros_like(general_func.flatten(normal_transformed)), 'o', label="original data: pca transformed")
        real_complex_transf_scatter = ax.plot(*zip(*complex_transformed.real), np.zeros_like(general_func.flatten(normal_transformed)), 'o', label=picked_options["pca_choice"].whoami() + ": transformed real")

        imag_transf_complex_arrows = []
        for i, p in enumerate(complex_transformed):
                    arrow = ax.plot(*zip(*[[p.real, 0], [p.real + p.imag, 0]]), label = label_name(picked_options["pca_choice"].whoami() + ": pca transformed imag", i), color="#696969", marker=">", markevery=[-1,-1])
                    imag_transf_complex_arrows.append(arrow)

    else:
        normal_transformed_scatter = ax.plot(general_func.flatten(normal_transformed), 'o', label="original data: pca transformed")

        real_complex_transf_scatter = ax.plot(*zip(*complex_transformed.real), 'o', label=picked_options["pca_choice"].whoami() + ": transformed real")
        imag_transf_complex_arrows = []
        for i, p in enumerate(complex_transformed):
            arrow = ax.plot(*zip(*[p.real, p.real + p.imag]), label = label_name(picked_options["pca_choice"].whoami() + ": pca transformed imag", i), color="#696969", marker=">", markevery=[-1,-1])
            imag_transf_complex_arrows.append(arrow)

        normal_transformed_vectors_arrows = []
        for i, p in enumerate(normal_vector_transformed):
            cmap = matplotlib.colormaps["Greens"]
            color_i = (len(normal_vector_transformed) - i) / len(normal_vector_transformed)
            rgba = cmap(color_i)
            arrow = ax.plot(*zip(*[normal_vector_transformed[i]]), 'o', label = label_name("vectors: pca transformed", i), color=rgba)
            normal_transformed_vectors_arrows.append(arrow)


    legend = ax.legend(loc='upper left')
    legends = legend.get_lines()

    for leg in legends:
        leg.set_picker(True)
        leg.set_pickradius(10)

    # normal_transf_scatter_legend, normal_transf_vector_legend, real_complex_transf_scatter_legend, trans_complex_vector_legend = legends
    if picked_options["target_dimension"] == 1:
        normal_transf_scatter_legend, real_complex_transf_scatter_legend,trans_complex_vector_legend, _ = legends
        graphs = {}
        graphs[normal_transf_scatter_legend] = [normal_transformed_scatter]
        graphs[real_complex_transf_scatter_legend] = [real_complex_transf_scatter]
        graphs[trans_complex_vector_legend] = imag_transf_complex_arrows
    else:
        normal_transf_scatter_legend, normal_transf_vector_legend, real_complex_transf_scatter_legend, trans_complex_vector_legend = legends

        graphs = {}
        graphs[normal_transf_scatter_legend] = [normal_transformed_scatter]
        graphs[normal_transf_vector_legend] = normal_transformed_vectors_arrows
        graphs[real_complex_transf_scatter_legend] = [real_complex_transf_scatter]
        graphs[trans_complex_vector_legend] = imag_transf_complex_arrows

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        try:
            graphs[legend].set_visible(not isVisible)
        except:
            for i in graphs[legend]:
                try:
                    for j in i:
                        j.set_visible(not isVisible)
                except:
                    i[0].set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()

    plt.connect('pick_event', on_pick)
    plt.title("Transformed data")
    ax.set_aspect('equal')
    plt.show()


def plot_pca_transf_with_id(picked_options):
    sep_points =  picked_options["sep_points"]
    order =  picked_options["order"]
    target_dimension =  picked_options["target_dimension"]
    own_PCA =  picked_options["pca_choice"]
    when_shuffle =  picked_options["when_shuffle"]

    flat_data = general_func.flatten(sep_points)
    data = []
    num_points = len(sep_points[0])
    separated = False

    sep_points, separated = generate_correct_ordered_data(order, num_points, picked_options["num_lines"], picked_options["starting_dimensions"], picked_options["jitter_bound"], picked_options["bounds"], picked_options["parallel_start_end"])

    vectors = get_neighbor_vector(data, separated)
    complex_data = add_complex_vector(data, separated)
    id_data = add_id(data, separated, normed = True)

    if separated:
        data = general_func.flatten(data)
        id_data = flatten(id_data)

    _, normal_transformed = pca_transform_data(data, target_dimension, own_PCA, None, when_shuffle)
    if picked_options["scaling_choice"] != "vector_pca":
        _, complex_transformed = pca_transform_data(complex_data, target_dimension, own_PCA, picked_options["scaling_choice"], when_shuffle)
    else:
        _, complex_transformed = vector_PCA.vector_pca_transform_data(data, vectors, target_dimension)
        
    _, id_transformed = pca_transform_data(id_data, target_dimension, own_PCA, None, when_shuffle)

    fig, ax = plt.subplots()
    dim = str(target_dimension) + 'd'
    try:
        ax = fig.add_subplot(projection=dim)
    except:
        pass

    if picked_options["target_dimension"] == 1:
        ax.plot(range(len(normal_transformed)), label=picked_options["order"])
        normal_transformed_scatter = ax.plot(general_func.flatten(normal_transformed), np.zeros_like(general_func.flatten(normal_transformed)), 'o', label="original data: pca transformed")
        real_complex_transf_scatter = ax.plot(*zip(*complex_transformed.real), np.zeros_like(general_func.flatten(normal_transformed)), 'o', label=picked_options["pca_choice"].whoami() + ": transformed real")

        imag_transf_complex_arrows = []
        for i, p in enumerate(complex_transformed):
                    arrow = ax.plot(*zip(*[[p.real, 0], [p.real + p.imag, 0]]), label = label_name(picked_options["pca_choice"].whoami() + ": pca transformed imag", i), color="#696969", marker=">", markevery=[-1,-1])
                    imag_transf_complex_arrows.append(arrow)    
    
    else:
        orig_transformed_scatter = ax.plot(*zip(*normal_transformed), 'o', label="original data: pca transformed")

        real_complex_transf_scatter = ax.plot(*zip(*complex_transformed.real), 'o', label=str(picked_options["pca_choice"].whoami()) + ": datapoints")

        imag_transf_complex_arrows = []
        for i, p in enumerate(complex_transformed):
            arrow = ax.plot(*zip(*[p.real, p.real + p.imag]), label = label_name(str(picked_options["pca_choice"].whoami()) + ": vectors", i), color="#696969", marker=">", markevery=[-1,-1])
            imag_transf_complex_arrows.append(arrow)
        
        id_list = []
        id_scatter_list = []
        for i,p in enumerate(id_transformed):
            cmap = matplotlib.colormaps["Purples"]
            color_i = (len(id_transformed) - i) / len(id_transformed)
            rgba = cmap(color_i)
            plot = ax.plot(*p, 'o', label = label_name("ID's", i), color= rgba)
            annot = ax.text(*p, str(round(p[target_dimension - 1], 2)))
            id_scatter_list.append(plot)
            id_list.append(annot)

    legend = ax.legend(loc='upper left')
    legends = legend.get_lines()
    for leg in legends:
        leg.set_picker(True)
        leg.set_pickradius(10)

    if picked_options["target_dimension"] == 1:
        normal_transf_scatter_legend, real_complex_transf_scatter_legend,trans_complex_vector_legend, _ = legends
        graphs = {}
        graphs[normal_transf_scatter_legend] = [normal_transformed_scatter]
        graphs[real_complex_transf_scatter_legend] = [real_complex_transf_scatter]
        graphs[trans_complex_vector_legend] = imag_transf_complex_arrows
    else:
        orig_legend, real_compl_legend, imag_compl_legend, id_legend = legends

        graphs = {}
        graphs[orig_legend] = [orig_transformed_scatter]
        graphs[real_compl_legend] = [real_complex_transf_scatter]
        graphs[imag_compl_legend] = imag_transf_complex_arrows
        graphs[id_legend] = [*id_scatter_list, id_list]

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        try:
            graphs[legend].set_visible(not isVisible)
        except:
            for i in graphs[legend]:
                try:
                    for j in i:
                        j.set_visible(not isVisible)
                except:
                    i[0].set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()

    plt.connect('pick_event', on_pick)
    plt.title("Transformed data")
    ax.set_aspect('equal')
    plt.show()


def compare_reconstr_err_order(real):
    target_dimension = 2
    num_lines = 4
    num_points = 20
    start_dim = 20

    re_zigzag = []
    re_compl_zigzag = []
    re_compl_zigzag_svd_real = []
    re_compl_zigzag_svd_complex = []
    re_one_line = []
    re_compl_one_line = []
    re_compl_one_line_svd_real = []
    re_compl_one_line_svd_complex = []

    for dimensions in range(2,start_dim,2):
        data = generate_data((0,100), dimensions, num_points, num_lines, 1)

        zigzag = orderings.zigzag_order(data, num_points)
        complex_zigzag = add_complex_vector(zigzag, False)

        one_line = orderings.one_line(data)
        complex_one_line = add_complex_vector(one_line, False)

        data = general_func.flatten(data)

        print("------------------------------------------------")
        print("---ZigZag---")
        print("Original")
        _, transf_zigzag = pca_transform_data(zigzag, target_dimension, complex_PCA.ComplexPCA, None, None)
        print("Complex")
        _, transf_compl_zigzag = pca_transform_data(complex_zigzag, target_dimension, complex_PCA.ComplexPCA, None, None)
        print("Method 1")
        _, transf_compl_zigzag_svd_real = pca_transform_data(complex_zigzag, target_dimension, complex_PCA.ComplexPCA_svd_real, None, None)
        print("Method 2")
        _, transf_compl_zigzag_svd_complex = pca_transform_data(complex_zigzag, target_dimension, complex_PCA.ComplexPCA_svd_complex, None, None)

        reconstr_zigzag, _ = pca_reconstruction(zigzag, transf_zigzag, target_dimension, complex_PCA.ComplexPCA, None, None)
        reconstr_compl_zigzag, _ = pca_reconstruction(zigzag, transf_compl_zigzag, target_dimension, complex_PCA.ComplexPCA, None, None)
        reconstr_complex_zigzag_svd_real, _ = pca_reconstruction(zigzag, transf_compl_zigzag_svd_real, target_dimension, complex_PCA.ComplexPCA_svd_real, None, None)
        reconstr_complex_zigzag_svd_complex, _ = pca_reconstruction(zigzag, transf_compl_zigzag_svd_complex, target_dimension, complex_PCA.ComplexPCA_svd_complex, None, None)
        
        print("---OneLine---")
        print("Original")
        _, transf_one_line = pca_transform_data(one_line, target_dimension, complex_PCA.ComplexPCA, None, None)
        print("Complex")
        _, transf_compl_one_line = pca_transform_data(complex_one_line, target_dimension, complex_PCA.ComplexPCA, None, None)
        print("Method 1")
        _, transf_compl_one_line_svd_real = pca_transform_data(complex_one_line, target_dimension, complex_PCA.ComplexPCA_svd_real, None, None)
        print("Method 2")
        _, transf_compl_one_line_svd_complex = pca_transform_data(complex_one_line, target_dimension, complex_PCA.ComplexPCA_svd_complex, None, None)

        reconstr_one_line, _ = pca_reconstruction(one_line, transf_one_line, target_dimension, complex_PCA.ComplexPCA, None, None)
        reconstr_compl_one_line, _ = pca_reconstruction(one_line, transf_compl_one_line, target_dimension, complex_PCA.ComplexPCA, None, None)
        reconstr_complex_one_line_svd_real, _ = pca_reconstruction(one_line, transf_compl_one_line_svd_real, target_dimension, complex_PCA.ComplexPCA_svd_real, None, None)
        reconstr_complex_one_line_svd_complex, _ = pca_reconstruction(one_line, transf_compl_one_line_svd_complex, target_dimension, complex_PCA.ComplexPCA_svd_complex, None, None)

        re_zigzag.append(reconstruction_error(zigzag, reconstr_zigzag, real))
        re_compl_zigzag.append(reconstruction_error(complex_zigzag, reconstr_compl_zigzag, real))
        re_compl_zigzag_svd_real.append(reconstruction_error(complex_zigzag, reconstr_complex_zigzag_svd_real, real))
        re_compl_zigzag_svd_complex.append(reconstruction_error(complex_zigzag, reconstr_complex_zigzag_svd_complex, real))
        re_one_line.append(reconstruction_error(one_line, reconstr_one_line, real))
        re_compl_one_line.append(reconstruction_error(complex_one_line, reconstr_compl_one_line, real))
        re_compl_one_line_svd_real.append(reconstruction_error(complex_one_line, reconstr_complex_one_line_svd_real, real))
        re_compl_one_line_svd_complex.append(reconstruction_error(complex_one_line, reconstr_complex_one_line_svd_complex, real))
    
    fig, ax = plt.subplots()
    re_zigzag_plot = ax.plot(range(2,start_dim,2), re_zigzag, label="PCA: original zigzag")
    re_compl_zigzag_plot = ax.plot(range(2,start_dim,2), re_zigzag, label="PCA: complex zigzag")
    re_compl_zigzag_svd_real_plot = ax.plot(range(2,start_dim,2), re_compl_zigzag_svd_real, label="svd real PCA: complex zigzag")
    re_compl_zigzag_svd_complex_plot = ax.plot(range(2,start_dim,2), re_compl_zigzag_svd_complex, label="svd complex PCA: complex zigzag")
    re_one_line_plot = ax.plot(range(2,start_dim,2), re_one_line, label="PCA: original one_line")
    re_compl_one_line_plot = ax.plot(range(2,start_dim,2), re_one_line, label="PCA: complex one_line")
    re_compl_one_line_svd_real_plot = ax.plot(range(2,start_dim,2), re_compl_one_line_svd_real, label="svd real PCA: complex one_line")
    re_compl_one_line_svd_complex_plot = ax.plot(range(2,start_dim,2), re_compl_one_line_svd_complex, label="svd complex PCA: complex one_line")
    
    legend = ax.legend(loc='lower right')
    legends = legend.get_lines()

    for leg in legends:
        leg.set_picker(True)
        leg.set_pickradius(10)

    re_zigzag_legend, re_compl_zigzag_legend, re_compl_zigzag_svd_real_legend, re_compl_zigzag_svd_complex_legend, re_one_line_legend, re_compl_one_line_legend, re_compl_one_line_svd_real_legend, re_compl_one_line_svd_complex_legend = legend.get_lines()
    
    graphs = {}
    graphs[re_zigzag_legend] = [re_zigzag_plot]
    graphs[re_compl_zigzag_legend] = [re_compl_zigzag_plot]
    graphs[re_compl_zigzag_svd_real_legend] = [re_compl_zigzag_svd_real_plot]
    graphs[re_compl_zigzag_svd_complex_legend] = [re_compl_zigzag_svd_complex_plot]
    graphs[re_one_line_legend] = [re_one_line_plot]
    graphs[re_compl_one_line_legend] = [re_compl_one_line_plot]
    graphs[re_compl_one_line_svd_real_legend] = [re_compl_one_line_svd_real_plot]
    graphs[re_compl_one_line_svd_complex_legend] = [re_compl_one_line_svd_complex_plot]

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        try:
            graphs[legend].set_visible(not isVisible)
        except:
            for i in graphs[legend]:
                try:
                    for j in i:
                        j.set_visible(not isVisible)
                except:
                    i[0].set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()

    plt.connect('pick_event', on_pick)
    plt.title("Reconstruction error for one line and zigzag ordered data: Comparison of different PCA variants")
    plt.xlabel("Starting dimension which is reduced to 2 dimensions")
    plt.ylabel("Reconstruction error")
    plt.show()


def compare_pca_transformed_by_order(picked_options):
    flat_data = general_func.flatten(picked_options["sep_points"])
    data = []
    num_points = len(picked_options["sep_points"][0])
    separated = False

    if picked_options["order"] == "zigzag":
        data = orderings.zigzag_order(picked_options["sep_points"], num_points)
    elif picked_options["order"] == "one_line":
        data = orderings.one_line(picked_options["sep_points"])
    elif picked_options["order"] == "random":
        data = orderings.random_ordering(flat_data)
    elif picked_options["order"] in ["swiss_roll", "spiral"]:
        data = picked_options["sep_points"]
    elif picked_options["order"] in ["sep_lines", "helices", "clusters", "staircase", "parallel_lines"]:
        separated = True
        data = picked_options["sep_points"]

    rev_data = orderings.reverse_ordering(data)
    vectors = get_neighbor_vector(data, separated)
    complex_data = add_complex_vector(data, separated)

    rev_complex_data = add_complex_vector(rev_data, separated)

    if separated:
        data = general_func.flatten(data)
    
    if len(picked_options["outliers"]) != 0:
        complex_data = np.vstack((complex_data, picked_options["outliers"]))
        data = np.vstack((data, picked_options["outliers"]))

    _, normal_transformed = pca_transform_data(data, picked_options["target_dimension"], picked_options["pca_choice"], None, picked_options["when_shuffle"])
    _, normal_vector_transformed = pca_transform_data(vectors, picked_options["target_dimension"], picked_options["pca_choice"], None, picked_options["when_shuffle"])
    _, complex_transformed = pca_transform_data(complex_data, picked_options["target_dimension"], picked_options["pca_choice"], picked_options["scaling_choice"], picked_options["when_shuffle"])
    _, rev_complex_transformed = pca_transform_data(rev_complex_data, picked_options["target_dimension"], picked_options["pca_choice"], picked_options["scaling_choice"], picked_options["when_shuffle"])

    return normal_transformed, normal_vector_transformed, complex_transformed, rev_complex_transformed


def compare_pca_reconstruction_by_order(picked_options):
    
    # Get the correct data structure
    sep_points = picked_options["sep_points"]
    separated = picked_options["separated"]

    complex_data = add_complex_vector(sep_points, separated)
    rev_data = orderings.reverse_ordering(sep_points)
    vectors = get_neighbor_vector(sep_points, separated)

    rev_complex_data = add_complex_vector(rev_data, separated)

    if separated:
        data = general_func.flatten(sep_points)
    
    if len(picked_options["outliers"]) != 0:
        complex_data = np.vstack((complex_data, picked_options["outliers"]))
        data = np.vstack((data, picked_options["outliers"]))

    normal_transformed, normal_vector_transformed, complex_transformed, rev_complex_transformed = compare_pca_transformed_by_order(picked_options)

    reconstr_normal_transformed, _ = pca_reconstruction(data, picked_options["target_dimension"], picked_options["pca_choice"], picked_options["scaling_choice"], picked_options["when_shuffle"])
    reconstr_vectors_transformed, _ = pca_reconstruction(vectors, picked_options["target_dimension"], picked_options["pca_choice"], picked_options["scaling_choice"],picked_options["when_shuffle"])
    if picked_options["scaling_choice"] != "vector_pca":
        reconstr_complex_transformed, _ = pca_reconstruction(complex_data, picked_options["target_dimension"], picked_options["pca_choice"], picked_options["scaling_choice"], picked_options["when_shuffle"])
    else:
        reconstr_complex_transformed, _ = vector_PCA.vector_pca_reconstruction(np.real(complex_data), np.imag(complex_data), picked_options["target_dimension"])
    reconstr_rev_complex_transformed, _ = pca_reconstruction(rev_complex_data, picked_options["target_dimension"], picked_options["pca_choice"], picked_options["scaling_choice"], picked_options["when_shuffle"])

    normal_reconstr_error = reconstruction_error(data, reconstr_normal_transformed, picked_options["rec_error_real"])
    complex_reconstr_error = reconstruction_error(complex_data, reconstr_complex_transformed, picked_options["rec_error_real"])
    rev_complex_error = reconstruction_error(rev_complex_data, reconstr_rev_complex_transformed, picked_options["rec_error_real"])

    print("----------------------------------------------------------")
    print(picked_options["order"])
    print("Normal PCA reconstruction error: " + str(normal_reconstr_error))
    print("Complex PCA reconstruction error: " + str(complex_reconstr_error))
    print("Complex PCA on reversed data reconstruction error: " + str(rev_complex_error))

    return (data, complex_data, reconstr_normal_transformed, reconstr_vectors_transformed, reconstr_complex_transformed, reconstr_rev_complex_transformed)


def plot_dim_red_timeseries(timeseries, target_dimension):
    complex_timeseries = add_complex_vector(timeseries, False)
    id_timeseries = add_id(timeseries, False)

    _, transf_meth1_complex_timeseries = pca_transform_data(complex_timeseries, target_dimension, complex_PCA.ComplexPCA_svd_real, None)
    _, transf_meth2_complex_timeseries = pca_transform_data(complex_timeseries, target_dimension, complex_PCA.ComplexPCA_svd_complex, None)
    _, transf_id_timeseries = pca_transform_data(id_timeseries, target_dimension, complex_PCA.ComplexPCA, None)

    fig, ax = plt.subplots()
    dim = str(target_dimension) + 'd'
    try:
        ax = fig.add_subplot(projection=dim)
    except:
        pass

    scatter_meth1 = ax.plot(*zip(*transf_meth1_complex_timeseries), 'o', label="Method1: real")

    imag_transf_complex_arrows = []
    for i, p in enumerate(transf_meth1_complex_timeseries):
        arrow = ax.plot(*zip(*[p.real, p.real + p.imag]), label = label_name("whole complex data: pca transformed imag", i), color="#696969", marker=">", markevery=[-1,-1])
        imag_transf_complex_arrows.append(arrow)
    
    scatter_meth2 = ax.plot(*zip(*transf_meth2_complex_timeseries), 'o', label="Method2: real")

    line_meth2_arrow = []
    for i, p in enumerate(transf_meth2_complex_timeseries):
        arrow = ax.plot(*zip(*[p.real, p.real + p.imag]), label = label_name("Method2: imaginary", i), color="#898989", marker=">", markevery=[-1,-1])
        line_meth2_arrow.append(arrow)
    
    scatter_id = ax.plot(*zip(*transf_id_timeseries), 'o', label = "ID's")
    
    legend = ax.legend(loc='upper left')
    legends = legend.get_lines()
    for leg in legends:
        leg.set_picker(True)
        leg.set_pickradius(10)

    meth1_real_legend, meth1_imag_legend, meth2_real_legend, meth2_imag_legend, id_legend = legends

    graphs = {}
    graphs[meth1_real_legend] = [scatter_meth1]
    graphs[meth1_imag_legend] = imag_transf_complex_arrows
    graphs[meth2_real_legend] = [scatter_meth2]
    graphs[meth2_imag_legend] = line_meth2_arrow
    graphs[id_legend] = [scatter_id]

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        try:
            graphs[legend].set_visible(not isVisible)
        except:
            for i in graphs[legend]:
                try:
                    for j in i:
                        j.set_visible(not isVisible)
                except:
                    i[0].set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()

    plt.connect('pick_event', on_pick)
    plt.title("Transformed data")
    ax.set_aspect('equal')
    plt.show()


def plot_structure(sep_points, order, start_dim, target_dimension):
    flat_data = general_func.flatten(sep_points)
    data = []
    num_points = len(sep_points[0])
    num_lines = len(sep_points)
    
    if order == "zigzag":
        data = orderings.zigzag_order(sep_points, num_points)
    elif order == "one_line":
        data = orderings.one_line(sep_points)
    elif order == "random":
        data = orderings.random_ordering(flat_data)
    if order == "sep_lines":
        data = sep_points
        
    complex_data = add_complex_vector(data, start_dim, num_lines, num_points)
    
    if order == "sep_lines":
        data = general_func.flatten(data)
    
    _, imag_transformed = pca_transform_data(complex_data.imag, target_dimension, complex_PCA.ComplexPCA, None)
    _, imagpc_complex_transformed = pca_transform_data(complex_data, target_dimension, complex_PCA.ComplexPCA_svd_complex_imag_pc, None)
    _, imagpc_imag_transformed = pca_transform_data(complex_data, target_dimension, complex_PCA.ComplexPCA_svd_complex_imag_pc_on_imag, None)
    
    fig, ax = plt.subplots()
    dim = str(target_dimension) + 'd'

    try:
        ax = fig.add_subplot(projection=dim)
    except:
        pass
    
    # PCA on only imaginary part
    transformed_vectors_arrows = []
    for i, p in enumerate(imag_transformed):
        cmap = matplotlib.colormaps["Greens"]
        color_i = (len(imag_transformed) - i) / len(imag_transformed)
        rgba = cmap(color_i)
        arrow = ax.plot(*zip(*[imag_transformed[i]]), 'o', label = label_name("Method 1: pca on imaginary only", i), color=rgba)
        transformed_vectors_arrows.append(arrow)
        
    imagpc_imag = []
    for i, p in enumerate(imagpc_imag_transformed):
        cmap = matplotlib.colormaps["Blues"]
        color_i = (len(imagpc_imag_transformed) - i) / len(imagpc_imag_transformed)
        rgba = cmap(color_i)
        arrow = ax.plot(*zip(*[imagpc_imag_transformed[i]]), 'o', label = label_name("Method 2: on only imaginary parts", i), color=rgba)
        imagpc_imag.append(arrow)
    
    legend = ax.legend(loc='upper right')
    legends = legend.get_lines()

    for leg in legends:
        leg.set_picker(True)
        leg.set_pickradius(10)

    transformed_vectors_arrows_legend, imagpc_imag_legend = legend.get_lines()
    graphs = {}
    
    graphs[transformed_vectors_arrows_legend] = transformed_vectors_arrows
    graphs[imagpc_imag_legend] = imagpc_imag

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        try:
            graphs[legend].set_visible(not isVisible)
        except:
            for i in graphs[legend]:
                try:
                    for j in i:
                        j.set_visible(not isVisible)
                except:
                    i[0].set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()

    plt.connect('pick_event', on_pick)
    plt.show()


def plot_pca_arrows(picked_options):
    fig = plt.figure()
    ax = fig.add_subplot()
    dim = str(picked_options["starting_dimension"]) + 'd'

    try:
        ax = fig.add_subplot(111, projection=dim)
    except:
        pass
    
    target_dim = 2
    start_dim  = 2
    order = "zigzag"
    num_points_per_line = 10
    num_lines = 2
    
    # Get data
    sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dim, 0, [0,100], picked_options["parallel_start_end"])
    
    if separated:
        data = flatten(sep_points)
    else:
        data = sep_points
    complex_data = add_complex_vector(data, separated)
    neighbours = get_neighbor_vector(sep_points, separated)
    
    pca = complex_PCA.ComplexPCA(picked_options["starting_dimension"], None, None)
    pca.fit(data)
    pca_loadings = pca.get_principal_loadings()
    
    vector_pca = VectorPCA(picked_options["starting_dimension"])
    vector_pca.fit(data, neighbours)
    vector_pca_loadings = vector_pca.get_principal_loadings()
    
    arrows = []
    for i,p in enumerate(complex_data):
        arrow = ax.plot(*zip(*[p.real, p.real + p.imag]), color="silver")
        arrows.append(arrow)
        
    plt.scatter(*zip(*data))
    
    mean = data.mean(axis = 0)
    
    for i,p in enumerate(pca_loadings):
        arrow = ax.plot(*zip(mean, mean + p), color="red")
        
    # for i,p in enumerate(vector_pca_loadings):
    #     arrow = ax.plot([mean, mean + p], color="gold")
    

    # pca_line_arrow_list = []
    # mean, v_list = complex_PCA.complex_pca(data, picked_options["starting_dimension"], complex_PCA.ComplexPCA)
    # for i in range(0, len(v_list)):
    #     pca_arrow = ax.plot(*zip(*[mean, (mean + v_list[i])]), label=label_name("pca", i), color = "g", marker=">", markevery=[-1,-1])
    #     pca_line_arrow_list.append(pca_arrow)
    
    # comp_pca_line_arrow_list = []
    # mean, v_list = complex_PCA.complex_pca(complex_data, picked_options["starting_dimension"], picked_options["pca_choice"], picked_options["scaling_choice"])
    # for i in range(0, len(v_list)):
    #     pca_arrow = ax.plot(*zip(*[mean, (mean + v_list[i])]), label=label_name("method 2", i), color = "r", marker=">", markevery=[-1,-1])
    #     comp_pca_line_arrow_list.append(pca_arrow)

    # orig_points = ax.plot(*zip(*data), 'o', label="Original datapoints")
    ax.set_aspect('equal', 'box')
    fig.tight_layout(pad = 5)
    plt.legend()
    plt.show()


def plot_rec_neighborh_error(order, parallel_start_end, sep_measure, dist_measure, bounds, dimensions, target_dimensions, jitter_bound, num_points, num_lines, max_k, variance, cut_k = None, scaling_choice = "vector_pca", sep_points = None):
    """
    Plot average deviation vs. increasing number of neighbours.

    Parameters:
        picked_options (dictionary): Input data matrix with shape (n_samples, n_features).
        k (int): Maximum number of dimensions to consider for dimensionality reduction.

    Returns:
        List of means and variances for pca, vector_pca and random_projection
    """
    if dist_measure == "multiple_scalar_product":
        variance = False
    assert dist_measure != "dtw", "Error DTW-measure does not work for this function"
    separated = False
    num_runs = 1
    if order == "russell1000_stock":
        bounds = [0.08, 2881.94]
        dimensions = 1598
        num_points = 504
    # elif order == "flights":
    #     bounds = [0.0, 83214.0]
    elif order == "air_pollution":
        if num_points == 8760:
            bounds = [-10000.0, 65535.0]
        elif num_points == 4380:
            bounds = [-918.0, 4379.0]
        
    parameters = [dimensions, target_dimensions, order, bounds, num_lines, num_points, dist_measure, max_k, sep_measure, jitter_bound, parallel_start_end, num_runs, scaling_choice]
    result_list = find_in_pickle_all_without_seed("./pickled_results/neighb_error.pickle", parameters)
    slurm_result_list = find_in_pickle_all_without_seed("./pickled_results/slurm/neighb_error.pickle", parameters)
    other_results = find_in_pickle_all_without_seed("./pickled_results/other_methods_neigh_error.pickle", parameters)
    slurm_other_results = find_in_pickle_all_without_seed("./pickled_results/slurm/other_methods_neigh_error.pickle", parameters)
    
    if len(result_list) != num_runs:

        result_list = slurm_result_list
    
    if len(other_results) != num_runs:
        other_results = slurm_other_results
    
    assert result_list is not None, "Error: There are no pickled results for pca, vector pca and random-projections"
    assert other_results is not None, "Error: There are no pickled results for lle, tsne and umap"
    
    res = result_list[0]
    other_res = other_results[0]
    pca_mean_list = res["pca_results"][0]
    pca_var_list = res["pca_results"][1]
    vector_mean_list = res["vector_pca_results"][0]
    vector_var_list = res["vector_pca_results"][1]
    random_proj_mean_list = res["rand_proj_results"][0]
    random_proj_var_list = res["rand_proj_results"][1]

    transf_pca_mean_list = res["transf_pca_results"][0]
    transf_pca_var_list = res["transf_pca_results"][1]
    transf_vector_mean_list = res["transf_vector_pca_results"][0]
    transf_vector_var_list = res["transf_vector_pca_results"][1]
    transf_random_proj_mean_list = res["transf_rand_proj_results"][0]
    transf_random_proj_var_list = res["transf_rand_proj_results"][1]
    
    transf_tsne_mean_list = other_res["lle_results"][0]
    transf_tsne_var_list = other_res["lle_results"][0]
    transf_lle_mean_list = other_res["tsne_results"][0]
    transf_lle_var_list = other_res["tsne_results"][0]
    transf_umap_mean_list = other_res["umap_results"][0]
    transf_umap_var_list = other_res["umap_results"][0]
    
    fig, (ax1, ax2) = plt.subplots(2, figsize=(14,8))

    # Plot for PCA for reconstructed data
    upper_var = np.array(pca_mean_list) + np.array(pca_var_list)
    lower_var = np.array(pca_mean_list) - np.array(pca_var_list)
    if sep_measure:
        ax1.plot(range(1, max_k + 1), pca_mean_list, label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
        if variance:
            ax1.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.4, color = "#007a33")
    else: 
        ax1.plot(range(1, max_k + 1), pca_mean_list, label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
        if variance:
            ax1.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.4, color = "#007a33", linewidth = 2)

    # Plot for random projection for reconstructed data
    upper_var = np.array(random_proj_mean_list) + np.array(random_proj_var_list)
    lower_var = np.array(random_proj_mean_list) - np.array(random_proj_var_list)

    if sep_measure:
        ax1.plot(range(1, max_k + 1), random_proj_mean_list, label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
        if variance:    
            ax1.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color = "#ff42c8")
    else: 
        ax1.plot(range(1, max_k + 1), random_proj_mean_list, label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
        if variance:    
            ax1.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color = "#ff42c8", linewidth = 2)
        
    # Plot for Vector PCA for reconstructed data
    upper_var = np.array(vector_mean_list) + np.array(vector_var_list)
    lower_var = np.array(vector_mean_list) - np.array(vector_var_list)

    if sep_measure:
        ax1.plot(range(1, max_k + 1), vector_mean_list, label="PCA*", color = "gold", linewidth=3)
        if variance:
            ax1.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color = "gold")
    else: 
        ax1.plot(range(1, max_k + 1), vector_mean_list, label="PCA*", color = "gold", linewidth=3)
        if variance:    
            ax1.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.6, color = "gold", linewidth = 2)

    
    # Plot for lle for transformed data
    upper_var = np.array(transf_lle_mean_list) + np.array(transf_lle_var_list)
    lower_var = np.array(transf_lle_mean_list) - np.array(transf_lle_var_list)

    if sep_measure:
        ax2.plot(range(1, max_k + 1), transf_lle_mean_list, label="LLE", linestyle="dotted", color="#e00031", linewidth=3)
        if variance:
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color="#e00031")
    else: 
        ax2.plot(range(1, max_k + 1), transf_lle_mean_list, label="LLE", linestyle="dotted", color="#e00031", linewidth=3)
        if variance:
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color="#e00031")
    
    # Plot for tsne for transformed data
    upper_var = np.array(transf_tsne_mean_list) + np.array(transf_tsne_var_list)
    lower_var = np.array(transf_tsne_mean_list) - np.array(transf_tsne_var_list)

    if sep_measure:
        ax2.plot(range(1, max_k + 1), transf_tsne_mean_list, label="t-SNE", marker=".", color="#9b63f3", linewidth=3)
        if variance:
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color="#9b63f3")
    else: 
        ax2.plot(range(1, max_k + 1), transf_tsne_mean_list, label="t-SNE", marker=".", color="#9b63f3", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color="#9b63f3")
    
    # Plot for umap for transformed data
    upper_var = np.array(transf_umap_mean_list) + np.array(transf_umap_var_list)
    lower_var = np.array(transf_umap_mean_list) - np.array(transf_umap_var_list)

    if sep_measure:
        ax2.plot(range(1, max_k + 1), transf_umap_mean_list, label="UMAP", marker="x", color="#2b9de9", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color="#2b9de9")
    else: 
        ax2.plot(range(1, max_k + 1), transf_umap_mean_list, label="UMAP", marker="x", color="#2b9de9", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color="#2b9de9")
        
    # Plot for random projection for transformed data
    upper_var = np.array(transf_random_proj_mean_list) + np.array(transf_random_proj_var_list)
    lower_var = np.array(transf_random_proj_mean_list) - np.array(transf_random_proj_var_list)

    if sep_measure:
        ax2.plot(range(1, max_k + 1), transf_random_proj_mean_list, label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color = "#ff42c8")
    else: 
        ax2.plot(range(1, max_k + 1), transf_random_proj_mean_list, label="Random Projection", color = "#ff42c8", linestyle="dashdot", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color = "#ff42c8", linewidth = 2)
    
    # Plot for PCA for transformed data
    upper_var = np.array(transf_pca_mean_list) + np.array(transf_pca_var_list)
    lower_var = np.array(transf_pca_mean_list) - np.array(transf_pca_var_list)

    if sep_measure:
        ax2.plot(range(1, max_k + 1), transf_pca_mean_list, label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color = "#007a33")
    else: 
        ax2.plot(range(1, max_k + 1), transf_pca_mean_list, label="PCA", color = "#007a33", linestyle="dashed", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.6, color = "#007a33", linewidth = 2)

    # Plot for PCA* for transformed data
    upper_var = np.array(transf_vector_mean_list) + np.array(transf_vector_var_list)
    lower_var = np.array(transf_vector_mean_list) - np.array(transf_vector_var_list)

    if sep_measure:
        ax2.plot(range(1, max_k + 1), transf_vector_mean_list, label="PCA*", color = "gold", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.3, color = "gold")
    else: 
        ax2.plot(range(1, max_k + 1), transf_vector_mean_list, label="PCA*", color = "gold", linewidth=3)
        if variance:    
            ax2.fill_between(range(1, max_k + 1), lower_var, upper_var, alpha = 0.6, color = "gold", linewidth = 2)
    fontsize = 33
    fontdist = 6
    ax1.set_xlabel("Number of neighbors", fontsize = fontsize - fontdist)
    ax1.set_ylabel("Avg. dev. \n from original data", fontsize = fontsize - fontdist)
    ax1.set_title("Reconstruction Result", fontdict = {"fontsize" : fontsize})
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    range_max = max(flatten([pca_mean_list, vector_mean_list, random_proj_mean_list]))
    if dist_measure not in ["multiple_scalar_product", "scalar_product"] and order != "flights":
        variance_max = max(flatten([pca_var_list, vector_var_list, random_proj_var_list]))
        range_max += variance_max
    next_higher = pow(10, len(str(round(range_max))))
    if range_max < next_higher - int(next_higher / 2):
            next_higher = next_higher - int(next_higher / 2)
    parts = 5
    ax1.set_yticks(range(0, next_higher, int(next_higher / parts)))
    if order == "flights":
        ax1.set_yscale("log")
        # ax1.set_yscale("asinh")
        # ax1.set_title("Log. Reconstruction Result", fontsize = fontsize)
    ax1.legend(loc= "upper left", bbox_to_anchor=(1.05, 0.8), borderaxespad=0, fontsize = fontsize - fontdist)
    ax1.tick_params(direction='out', labelsize=fontsize - fontdist)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.set_ylim(0)
    ax1.grid(axis='y', linestyle="dashed", color="silver", linewidth=1.5)
    
    ax2.set_xlabel("Number of neighbors", fontsize = fontsize - fontdist)
    ax2.set_ylabel("Avg. dev. \n from original data", fontsize = fontsize - fontdist)
    ax2.set_title("Transformation Result", fontdict = {"fontsize" : fontsize})
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax2.set_ylim(0)
    if order == "flights":
        ax1.set_yscale("log") 
    #     ax2.set_title("Log. Transformation", fontsize = fontsize)
    ax2.legend(loc= "upper left", bbox_to_anchor=(1.05, 0.8), borderaxespad=0, fontsize = fontsize - fontdist)
    ax2.tick_params(direction='out', labelsize=fontsize - fontdist)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.grid(axis='y', linestyle="dashed", color="silver", linewidth=1.5)
    range_max = max(flatten([transf_pca_mean_list, transf_vector_mean_list, transf_random_proj_mean_list, transf_lle_mean_list, transf_tsne_mean_list, transf_umap_mean_list]))
    if dist_measure not in ["multiple_scalar_product", "scalar_product"]:
        variance_max = max(flatten([transf_pca_var_list, transf_vector_var_list, transf_random_proj_var_list, transf_lle_var_list, transf_tsne_var_list, transf_umap_var_list]))
        range_max += variance_max
    next_higher = pow(10, len(str(round(range_max))))
    if range_max < next_higher - int(next_higher / 2):
            next_higher = next_higher - int(next_higher / 2)
    parts = 5
    ax2.set_yticks(range(0, next_higher, int(next_higher / parts)))
        
    # plt.suptitle("Order " + order + " measure " + str(k) + " neighbours for " + str(num_lines) + " lines a " + str(num_points) + " and jitter " + str(jitter_bound))
    fig.tight_layout()
    fig.set_figwidth(14)
    fig.set_figheight(8)
    # plt.box(False)
    if order in ["CMAPSS", "flights", "air_pollution", "russell2000_stock"]:
        plt.savefig("./plots/real-world/" + str(order) + "/num_neigh/transformed_" + str(num_lines) + "lines_" + str(num_points) + "points_" + str(dist_measure) + ".pdf", bbox_inches="tight")
    else:
        plt.savefig("./plots/generated/multiple_runs/" + str(order) + "/neighb_error/" + str(num_lines) + "lines_" + str(num_points) + "points_" + str(cut_k) + "neighbours_" + str(dist_measure) + ".pdf", bbox_inches="tight")

    plt.show()


def plot_transf_neighborh_error(picked_options, k, sep_measure):
    fig, ax = plt.subplots()
    
   # Get the correct data structure
    sep_points = picked_options["sep_points"]
    separated = picked_options["separated"]

    complex_data = add_complex_vector(sep_points, separated)
    id_data = add_id(sep_points, separated)

    if separated:
        data1 = flatten(sep_points)
        id_data = flatten(id_data)
    else:
        data1 = sep_points
        id_data = id_data
    
    if len(picked_options["outliers"]) != 0:
        complex_data = np.vstack((complex_data, picked_options["outliers"]))
        data1 = np.vstack((data1, picked_options["outliers"]))
    
    # Transform data
    _, transf_data = pca_transform_data(data1, picked_options["target_dimension"], complex_PCA.ComplexPCA, picked_options["scaling_choice"], None)

    if picked_options["scaling_choice"] != "vector_pca":
        _, transf_comp_data = pca_transform_data(complex_data, picked_options["target_dimension"], complex_PCA.ComplexPCA_svd_complex, picked_options["scaling_choice"], None)
    else:
        _, transf_comp_data = vector_PCA.vector_pca_transform_data(np.real(complex_data), np.imag(complex_data), picked_options["target_dimension"])

    _, transf_id_data = pca_transform_data(id_data, picked_options["target_dimension"], complex_PCA.ComplexPCA, None, None)

    # Add structure name to legend
    plt.plot([], [], ' ', label="structure " + str(picked_options["order"]))

    # Compute mean and std for data transformed by original pca, complex pca and with IDs
    if not sep_measure:
        exp_values, vars, median = get_mu_var_by_neighbor_num(data1, transf_data, picked_options["dist_measure"], k)
        exp_comp_values, comp_vars, comp_median = get_mu_var_by_neighbor_num(data1, transf_comp_data.real, picked_options["dist_measure"], k)
        id_exp_values, id_vars, id_median = get_mu_var_by_neighbor_num(data1, transf_id_data, picked_options["dist_measure"], k)
    elif sep_measure:
        sep_transf_data = [transf_data[i:i + picked_options["num_points"]] for i in range(0, len(transf_data), picked_options["num_points"])]
        sep_id_transf_data = [transf_id_data[i:i + picked_options["num_points"]] for i in range(0, len(transf_id_data), picked_options["num_points"])]
        sep_complex_transf_data = [transf_comp_data[i:i + picked_options["num_points"]] for i in range(0, len(transf_comp_data), picked_options["num_points"])]

        exp_values, vars, median = measure_sep_structures(picked_options["sep_points"], sep_transf_data, picked_options["dist_measure"], k)
        exp_comp_values, comp_vars, comp_median = measure_sep_structures(picked_options["sep_points"], sep_complex_transf_data, picked_options["dist_measure"], k)
        id_exp_values, id_vars, id_median = measure_sep_structures(picked_options["sep_points"], sep_id_transf_data, picked_options["dist_measure"], k)

    # Plot mean and std for orginal pca
    upper_var = np.array(exp_values) + np.array(vars)
    lower_var = np.array(exp_values) - np.array(vars)

    if not sep_measure:
        plt.plot(range(min(k, len(data1))), median, label="Median: Original", color = "c", linestyle="dotted")
        plt.plot(range(min(k, len(data1))), exp_values, label="Expected Value: Original", color = "c")
        plt.fill_between(range(min(k, len(data1))), lower_var, upper_var, alpha = 0.6, color="c")
    elif sep_measure:
        plt.plot(range(min(k, picked_options["num_points"])), median, label="Median: Original", color = "c", linestyle="dotted")
        plt.plot(range(min(k, picked_options["num_points"])), exp_values, label="Expected Value: Original", color = "c")
        plt.fill_between(range(min(k, picked_options["num_points"])), lower_var, upper_var, alpha = 0.6, color="c")

    # Plot mean and std for complex pca
    comp_lower_var = np.array(exp_comp_values) - np.array(comp_vars)
    comp_upper_var = np.array(exp_comp_values) + np.array(comp_vars)

    if not sep_measure:
        plt.plot(range(min(k, len(data1))), comp_median, label="Median: Method 2", color="orange", linestyle="dotted")
        plt.plot(range(min(k, len(data1))), exp_comp_values, label="Expected Value: Method 2", color="orange")
        plt.fill_between(range(min(k, len(data1))), comp_lower_var, comp_upper_var, alpha = 0.3, color = "orange")
    elif sep_measure:
        plt.plot(range(min(k, picked_options["num_points"])), comp_median, label="Median: Method 2", color="orange", linestyle="dotted")
        plt.plot(range(min(k, picked_options["num_points"])), exp_comp_values, label="Expected Value: Method 2", color="orange")
        plt.fill_between(range(min(k, picked_options["num_points"])), comp_lower_var, comp_upper_var, alpha = 0.3, color = "orange")
    
    # # Plot mean and std for ID data
    # id_lower_var = np.array(id_exp_values) - np.array(id_vars)
    # id_upper_var = np.array(id_exp_values) + np.array(id_vars)

    # if not sep_measure:
    #     plt.plot(range(min(k, len(data1))), id_median, label="Median: Method 2", color="red", linestyle="dotted")
    #     plt.plot(range(min(k, len(data1))), id_exp_values, label="Expected Value: Method 2", color="red")
    #     plt.fill_between(range(min(k, len(data1))), id_lower_var, id_upper_var, alpha = 0.3, color = "red")
    # elif sep_measure:
    #     plt.plot(range(min(k, picked_options["num_points"])), id_median, label="Median: Method 2", color="red", linestyle="dotted")
    #     plt.plot(range(min(k, picked_options["num_points"])), id_exp_values, label="Expected Value: Method 2", color="red")
    #     plt.fill_between(range(min(k, picked_options["num_points"])), id_lower_var, id_upper_var, alpha = 0.3, color = "red")

    ax.legend(loc='upper left')

    plt.title("Structure Preservation: Mean & Variance in transformed dim space")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Average deviation from original")
    plt.show()


def plot_reconstructed_distance_mat_heatmap(picked_options):
    separated = False

    order = picked_options["order"]
    sep_points = picked_options["sep_points"]

   # Get the correct data structure
    sep_points = picked_options["sep_points"]
    separated = picked_options["separated"]
    
    if separated:
        data1 = flatten(sep_points)
    else:
        data1 = sep_points
    
    data, _, rec_data, rec_vec, rec_complex, _ = compare_pca_reconstruction_by_order(picked_options)

    dist_mat = distance_matrix(data, data)
    rec_dist_mat = distance_matrix(rec_data, rec_data)
    rec_complex_dist_mat = distance_matrix(rec_complex, rec_complex)

    fig, axes = plt.subplots(2,2)

    axes[0, 0].imshow(dist_mat, cmap="hot", interpolation="nearest")
    axes[0, 0].set_title("Original distances")

    axes[1,0].imshow(rec_dist_mat, cmap="hot", interpolation="nearest")
    axes[1,0].set_title("Original PCA distances")

    axes[1,1].imshow(rec_complex_dist_mat, cmap="hot", interpolation="nearest")
    axes[1,1].set_title(str(picked_options["pca_choice"].whoami()) + " distances")

    fig.suptitle("Original space")
    plt.tight_layout()
    plt.show()


def plot_transformed_distance_mat_heatmap(picked_options):
    separated = False

    # Get the correct data structure
    sep_points = picked_options["sep_points"]
    separated = picked_options["separated"]
    
    if separated:
        data1 = flatten(sep_points)
    else:
        data1 = sep_points
    
    transf_data, _, transf_complex, _ = compare_pca_transformed_by_order(picked_options)

    dist_mat = distance_matrix(data1, data1)
    transf_dist_mat = distance_matrix(transf_data, transf_data)
    transf_complex_dist_mat = distance_matrix(transf_complex, transf_complex)

    fig, axes = plt.subplots(2,2)

    axes[0, 0].imshow(dist_mat, cmap="hot", interpolation="nearest")
    axes[0, 0].set_title("Original distances")

    axes[1,0].imshow(transf_dist_mat, cmap="hot", interpolation="nearest")
    axes[1,0].set_title("Original PCA transformed distances")

    axes[1,1].imshow(transf_complex_dist_mat, cmap="hot", interpolation="nearest")
    axes[1,1].set_title(str(picked_options["pca_choice"].whoami()) + " transformed distances")

    fig.suptitle("Transformed space")
    plt.tight_layout()
    plt.show()

def box_plot_scalar_product(order, start_dim, target_dim, num_lines, num_points_per_line, bounds, jitter, parallel_start_end, seed):
    already_computed = find_in_pickle_with_param("./pickled_results/scalar_products.pickle", [seed, start_dim, target_dim, order, bounds, num_lines, num_points_per_line, jitter, parallel_start_end])

    if already_computed is None:
        multiple_runs_scalar_product(order, start_dim, target_dim, num_lines, num_points_per_line, 1, bounds, jitter, parallel_start_end, seed = seed)

        already_computed = find_in_pickle_with_param("./pickled_results/scalar_products.pickle", [seed, start_dim, target_dim, order, bounds, num_lines, num_points_per_line, jitter, parallel_start_end])

    scal_prod_rec_pca = already_computed["rec_pca_scal_prods"]
    scal_prod_rec_vector_pca = already_computed["rec_vector_pca_scal_prods"]
    scal_prod_transf_pca = already_computed["transf_pca_scal_prods"]
    scal_prod_transf_vector_pca = already_computed["transf_vector_pca_scal_prods"]
    
    fig, axes = plt.subplots(2, 1)

    sns.boxplot(data = [scal_prod_rec_pca,scal_prod_rec_vector_pca], ax = axes[0])

    sns.boxplot(data = [scal_prod_transf_pca, scal_prod_transf_vector_pca], ax = axes[1])

    fig.suptitle("Scalar product measure for PCA and Vector PCA for " + str(num_lines) + " number of lines a " + str(num_points_per_line) + " points for " + str(seed) + " seed")
    plt.title("Data is in " + str(order) + " in dimension " + str(start_dim) + " and transformed to " + str(target_dim))

    plt.tight_layout()
    plt.show()


def box_plot_multiple_scalar_product(order, start_dim, target_dim, num_lines, num_points_per_line, bounds, jitter, parallel_start_end, num_runs):

    already_computed = find_in_pickle_all_without_seed("./pickled_results/scalar_products.pickle", [start_dim, target_dim, order, bounds, num_lines, num_points_per_line, jitter, parallel_start_end])

    if already_computed is not None:
        if num_runs > len(already_computed):
            missing_runs = num_runs - len(already_computed)

            multiple_runs_scalar_product(order, start_dim, target_dim, num_lines, num_points_per_line, missing_runs, bounds, jitter, parallel_start_end)

            already_computed = find_in_pickle_all_without_seed("./pickled_results/scalar_products.pickle", [start_dim, target_dim, order, bounds, num_lines, num_points_per_line, jitter, parallel_start_end])
        else:
            already_computed = already_computed[len(already_computed) - num_runs:]
    else:
        missing_runs = num_runs
        multiple_runs_scalar_product(order, start_dim, target_dim, num_lines, num_points_per_line, missing_runs, bounds, jitter, parallel_start_end)

        already_computed = find_in_pickle_all_without_seed("./pickled_results/scalar_products.pickle", [start_dim, target_dim, order, bounds, num_lines, num_points_per_line, jitter, parallel_start_end])

    scal_prod_rec_pca = []
    scal_prod_rec_vector_pca = []
    scal_prod_transf_pca = []
    scal_prod_transf_vector_pca = []
    
    for result in already_computed:
        scal_prod_rec_pca.append(np.mean(result["rec_pca_scal_prods"]) / np.max(result["rec_pca_scal_prods"] + result["rec_vector_pca_scal_prods"]))
        scal_prod_rec_vector_pca.append(np.mean(result["rec_vector_pca_scal_prods"]) / np.max(result["rec_pca_scal_prods"] + result["rec_vector_pca_scal_prods"]))
        scal_prod_transf_pca.append(np.mean(result["transf_pca_scal_prods"]))
        scal_prod_transf_vector_pca.append(np.mean(result["transf_vector_pca_scal_prods"]))

    # fig, axes = plt.subplots(2, 1)

    # sns.boxplot(data = [scal_prod_rec_pca,scal_prod_rec_vector_pca], ax = axes[0]).set_xticklabels(["PCA", "Vector PCA"])
    # sns.boxplot(data = [scal_prod_transf_pca, scal_prod_transf_vector_pca], ax = axes[1]).set_xticklabels(["PCA", "Vector PCA"])
    sns.boxplot(data = [scal_prod_rec_pca,scal_prod_rec_vector_pca]).set_xticklabels(["PCA", "Vector PCA"])
    if num_runs > 1:
        plt.suptitle("Scalar product measure for PCA and Vector PCA for " + str(num_lines) + " lines a " + str(num_points_per_line) + " datapoints with jitter " + str(jitter) + " and " + str(num_runs) + " runs" "\n"
        "Data is in " + str(order) + " order in dimension " + str(start_dim) + " and transformed to " + str(target_dim))

    plt.tight_layout()
    plt.savefig("./plots/scalar_prod_" + str(order) + "_dim" + str(start_dim) + "-" + str(target_dim) + "_jitter" + str(jitter) + "_numruns" + str(num_runs))
    plt.show()
