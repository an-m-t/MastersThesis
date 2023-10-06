import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from vector_PCA import *
from complex_PCA import *
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from data_generation import *
from general_func import flatten

def generate_s_data(num_points):
    x_list = []
    y_list = []
    
    radius = 1.0
    center_x = 0.0
    center_y = -1.0  # Shift the center downward to generate points on the lower half

    # Define the parameter values (angle from 0 to π for the lower half)
    theta = np.linspace(0, np.pi, int(num_points/ 2))

    # Calculate the x and y coordinates of the points on the lower half
    x = center_x + (radius - 0.75) * np.cos(theta)
    y = center_y + (radius) * np.sin(theta)
    x_list.extend(x[::-1])
    y_list.extend(y)
    
    radius = 1.0
    center_x = 0.5
    center_y = -1.0
    
    # Calculate the x and y coordinates of the points on the lower half
    x = center_x + (radius - 0.75) * np.cos(theta)
    y = center_y + radius * -np.sin(theta)
    
    x_list.extend(x[::-1])
    y_list.extend(y)
    
    z = [random.uniform(0, 3) for i in range(2000)]
    
    data = []
    for i in range(num_points):
        point = [x_list[i], y_list[i], z[i]]
        data.append(point)
    return np.array(data)


def plot_s_data(num_points):
    data  = generate_s_data(num_points)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(num_points):
        cmap = matplotlib.colormaps["gnuplot2"]
        color_i = (num_points - i) / num_points
        rgba = cmap(color_i)
        arrow = ax.plot(data[i][0], data[i][1], data[i][2], 'o', color=rgba)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(direction='out', labelsize=16)
    cax = ax.figure.axes[-1]

    # Create a scatter plot of the points on the lower half of the circle
    plt.title('Original data', fontsize = 26)
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling of the axes
    plt.show()

def plot_s_pca_transformation(num_points):
    data  = generate_s_data(num_points)
    _, data = pca_reconstruction(data, 2, "complex_pca", None, None)
    
    fig = plt.figure()
    # ax = plt.axes(projection='3d')
    ax = plt.axes()
    for i in range(num_points):
        cmap = matplotlib.colormaps["gnuplot2"]
        color_i = (num_points - i) / num_points
        rgba = cmap(color_i)
        arrow = ax.plot(data[i][0], data[i][1], 'o', color=rgba)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(direction='out', labelsize=16)
    cax = ax.figure.axes[-1]

    # Create a scatter plot of the points on the lower half of the circle
    plt.title('Transformation with PCA', fontsize = 26)
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling of the axes
    plt.show()
    

def plot_s_vector_pca_transformation(num_points):
    data  = generate_s_data(num_points)
    neighbours = get_neighbor_vector(data, False)
    _, data = vector_pca_transform_data(data, neighbours, 2)
    
    fig = plt.figure()
    # ax = plt.axes(projection='3d')
    ax = plt.axes()
    for i in range(num_points):
        cmap = matplotlib.colormaps["gnuplot2"]
        color_i = (num_points - i) / num_points
        rgba = cmap(color_i)
        arrow = ax.plot(data[i][0], data[i][1], 'o', color=rgba)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(direction='out', labelsize=16)
    cax = ax.figure.axes[-1]

    # Create a scatter plot of the points on the lower half of the circle
    plt.title('Transformation with PCA*', fontsize = 26)
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling of the axes
    plt.show()

def plot_s_lle_transformation(num_points):
    data  = generate_s_data(num_points)
    lle = LocallyLinearEmbedding(n_components=2,  eigen_solver="dense", n_neighbors=13)
    data = lle.fit_transform(data)
    
    fig = plt.figure()
    # ax = plt.axes(projection='3d')
    ax = plt.axes()
    for i in range(num_points):
        cmap = matplotlib.colormaps["gnuplot2"]
        color_i = (num_points - i) / num_points
        rgba = cmap(color_i)
        arrow = ax.plot(data[i][0], data[i][1], 'o', color=rgba)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(direction='out', labelsize=15)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=16)

    # Create a scatter plot of the points on the lower half of the circle
    # fig.tight_layout(pad=1)
    ax.set_aspect('equal', 'box')
    plt.title('Transformation with LLE', fontsize = 30)
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling of the axes
    plt.show()
    

def plot_s_umap_transformation(num_points):
    data  = generate_s_data(num_points)
    # tsne = TSNE(n_components=2, method="exact")
    # data = tsne.fit_transform(data)
    umap_reducer = UMAP(n_components=2, n_neighbours=5)
    scaled_data = StandardScaler().fit_transform(data)
    data = umap_reducer.fit_transform(scaled_data)
    fig = plt.figure()
    # ax = plt.axes(projection='3d')
    ax = plt.axes()
    for i in range(num_points):
        cmap = matplotlib.colormaps["gnuplot2"]
        color_i = (num_points - i) / num_points
        rgba = cmap(color_i)
        arrow = ax.plot(data[i][0], data[i][1], 'o', color=rgba)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(direction='out', labelsize=15)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=16)

    # Create a scatter plot of the points on the lower half of the circle
    plt.title('Transformation with UMAP', fontsize = 26)
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling of the axes
    plt.show()



def generate_wave_data(num_points):
    x_list = [random.uniform(0, 10) for i in range(num_points)]
    x_list.sort()
    y_list = [-np.cos(5*x)*1/2 for x in x_list]
    
    z = [random.uniform(0, 3)] * num_points
    
    data = []
    for i in range(num_points):
        point = [x_list[i], y_list[i], z[i]]
        data.append(point)
    return np.array(data)


def plot_wave_data(num_points):
    data  = generate_wave_data(num_points)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(num_points):
        cmap = matplotlib.colormaps["gnuplot2"]
        color_i = (num_points - i) / num_points
        rgba = cmap(color_i)
        arrow = ax.plot(data[i][0], data[i][1], data[i][2], 'o', color=rgba)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(direction='out', labelsize=15)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=16)

    # Create a scatter plot of the points on the lower half of the circle
    plt.title('Original data', fontsize = 32)
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling of the axes
    plt.show()


def plot_wave_pca_transformation(num_points):
    data  = generate_wave_data(num_points)
    _, data = pca_reconstruction(data, 1, "complex_pca", None, None)
    
    fig = plt.figure()
    # ax = plt.axes(projection='3d')
    ax = plt.axes()
    for i in range(num_points):
        cmap = matplotlib.colormaps["gnuplot2"]
        color_i = (num_points - i) / num_points
        rgba = cmap(color_i)
        arrow = ax.plot(data[i][0], 'o', color=rgba)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(direction='out', labelsize=15)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=16)

    # Create a scatter plot of the points on the lower half of the circle
    plt.title('Transformation with PCA', fontsize = 26)
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling of the axes
    plt.show()
    
    
def plot_wave_vector_pca_transformation(num_points):
    data  = generate_wave_data(num_points)
    neighbours = get_neighbor_vector(data, False)
    _, data = vector_pca_transform_data(data, neighbours, 1)
    
    fig = plt.figure()
    # ax = plt.axes(projection='3d')
    ax = plt.axes()
    for i in range(num_points):
        cmap = matplotlib.colormaps["Greens"]
        color_i = (num_points - i) / num_points
        rgba = cmap(color_i)
        arrow = ax.plot(data[i][0], 'o', color=rgba)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(direction='out', labelsize=15)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=16)

    # Create a scatter plot of the points on the lower half of the circle
    plt.title('Transformation with PCA*', fontsize = 26)
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling of the axes
    plt.show()

# plot_wave_data(1000)
# plot_wave_pca_transformation(100)
# plot_wave_vector_pca_transformation(100)


def plot_lines_data(order, num_points_per_line, num_lines, bounds):
    sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, 3, 0, bounds, True, False)
    if separated:
        data = flatten(sep_points)
    else: 
        data = sep_points
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    neighbours = get_neighbor_vector(sep_points, separated)
    for i in range(len(neighbours)):
        ax.plot(*zip(data[i], data[i] + neighbours[i]), color="gray")
        
    for i in range(num_points_per_line * num_lines):
        cmap = matplotlib.colormaps["magma"]
        color_i = (num_points_per_line * num_lines - i) / (num_points_per_line * num_lines)
        rgba = cmap(color_i)
        arrow = ax.plot(data[i][0], data[i][1], data[i][2], 'o', color=rgba)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(direction='out', labelsize=15)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=16)

    # Create a scatter plot of the points on the lower half of the circle
    plt.title('Original data', fontsize = 26)
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling of the axes
    plt.show()


def plot_lines_pca_transformation(order, num_points_per_line, num_lines, bounds, target_dim):
    sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, 3, 0, bounds, True, False)
    if separated:
        data = flatten(sep_points)
    else: 
        data = sep_points
    _, data = pca_reconstruction(data, target_dim, "complex_pca", None, None)
    
    fig = plt.figure()
    ax = plt.axes()
    for i in range(num_points_per_line * num_lines):
        cmap = matplotlib.colormaps["magma"]
        color_i = (num_points_per_line * num_lines - i) / (num_points_per_line * num_lines)
        rgba = cmap(color_i)
        if target_dim == 1:
            arrow = ax.plot(data[i][0], 'o', color=rgba)
        if target_dim == 2:
            arrow = ax.plot(data[i][0], data[i][1], 'o', color=rgba)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(direction='out', labelsize=15)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=16)

    # Create a scatter plot of the points on the lower half of the circle
    plt.title('Transformation with PCA', fontsize = 26)
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling of the axes
    plt.show()
    
    
def plot_lines_vector_pca_transformation(order, num_points_per_line, num_lines, bounds, target_dim):
    sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, 3, 0, bounds, True, False)
    if separated:
        data = flatten(sep_points)
    else: 
        data = sep_points
    neighbours = get_neighbor_vector(data, False)
    _, data = vector_pca_transform_data(data, neighbours, target_dim)
    
    fig = plt.figure()
    # ax = plt.axes(projection='3d')
    ax = plt.axes()
    for i in range(num_points_per_line * num_lines):
        cmap = matplotlib.colormaps["magma"]
        color_i = (num_points_per_line * num_lines - i) / (num_points_per_line * num_lines)
        rgba = cmap(color_i)
        if target_dim == 1:
            arrow = ax.plot(data[i][0], 'o', color=rgba)
        if target_dim == 2:
            arrow = ax.plot(data[i][0], data[i][1], 'o', color=rgba)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.tick_params(direction='out', labelsize=15)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=16)

    # Create a scatter plot of the points on the lower half of the circle
    plt.title('Transformation with PCA*', fontsize = 26)
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling of the axes
    plt.show()

# num_points = 100
# num_lines = 2
# bounds = [0, 10]
# target_dim = 1
# order = "zigzag"
# seed = 123

# random.seed(seed)
# np.random.seed(seed)

def plot_demo_pca_arrows(seed, order):
    starting_dimension = 2
    random.seed(seed)
    np.random.seed(seed)
    fig = plt.figure()
    ax = fig.add_subplot()
    dim = str(starting_dimension) + 'd'

    try:
        ax = fig.add_subplot(111, projection=dim)
    except:
        pass
    target_dim = 2
    start_dim  = 2
    num_points_per_line = 10
    num_lines = 2
    
    # Get data
    sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dim, 0, [0,100], False)
    
    if separated:
        data = flatten(sep_points)
    else:
        data = sep_points
    complex_data = add_complex_vector(data, separated)
    neighbours = get_neighbor_vector(sep_points, separated)
    
    pca = complex_PCA.ComplexPCA(starting_dimension, None, None)
    pca.fit(data)
    pca_loadings = pca.get_principal_loadings()
    
    vector_pca = VectorPCA(starting_dimension)
    vector_pca.fit(data, neighbours)
    vector_pca_loadings = vector_pca.get_principal_loadings()
    
    # Plot ordering
    arrows = []
    for i,p in enumerate(complex_data):
        arrow = ax.plot(*zip(*[p.real, p.real + p.imag]), color="silver", zorder = 0)
        arrows.append(arrow)
        
    # Plot data    
    ax.plot(*zip(*data), 'o', markersize=14, color="purple")
    
    # Find starting points for pc 
    mean = data.mean(axis = 0)
    
    # Plot PCA pcs
    for i,p in enumerate(pca_loadings):
        arrow = plt.arrow(*mean, *p, color="#007a33", width = 0.8, length_includes_head = True, head_width = 3, head_length = 0.7, label = general_func.label_name("PCA", i), zorder= 5)
        
    # Plot Vector PCA pcs    
    for i,p in enumerate(vector_pca_loadings):
        arrow = plt.arrow(*mean, *p, color="gold", width = 1.3, length_includes_head = True, head_width = 3.5, head_length = 0.7, label = general_func.label_name("PCA*", i), zorder = 5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(direction='out', labelsize=15)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=16)
    ax.set_aspect('equal')
    plt.grid(True)
    ax.set(xlim=(0,100), ylim=(0,100))
    fig.tight_layout()
    plt.legend()
    plt.show()
    
    
def plot_demo_orders(seed, order):
    starting_dimension = 2
    random.seed(seed)
    np.random.seed(seed)
    fig = plt.figure()
    ax = fig.add_subplot()
    dim = str(starting_dimension) + 'd'

    try:
        ax = fig.add_subplot(111, projection=dim)
    except:
        pass
    target_dim = 2
    start_dim  = 2
    num_points_per_line = 10
    num_lines = 2
    
    # Get data
    sep_points, separated = generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dim, 0, [0,100], False)
    
    if separated:
        data = flatten(sep_points)
    else:
        data = sep_points
    complex_data = add_complex_vector(sep_points, separated)
    print(complex_data)
    # Plot ordering
    arrows = []
    for i,p in enumerate(complex_data):
        arrow = ax.plot(*zip(*[p.real, p.real + p.imag]), color="gray", zorder = 0, linewidth = 4)
        arrows.append(arrow)
        
    # Plot data    
    ax.plot(*zip(*data), 'o', markersize=14, color="purple")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(direction='out', labelsize=15)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=16)
    ax.set_aspect('equal')
    plt.grid(True)
    ax.set(xlim=(0,100), ylim=(0,100))
    fig.tight_layout()
    plt.show()