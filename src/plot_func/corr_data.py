from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from real_PCA import pca
import pandas as pd
import numpy as np

# Generate correlated dataset
def corr_data(num_points):
    cov = np.array([[1, 0.8],[1,1]])
    scores = mvn.rvs(mean = [50, 50], cov=cov, size = num_points)
    df = pd.DataFrame(data = 15 * scores + 60)
    correlated = df.values.tolist()

    return correlated


def plot_corr_points(points, dimensions):
    dim = str(dimensions) + 'd'

    fig = plt.figure()
    ax = fig.add_subplot()

    try:
        ax = fig.add_subplot(projection=dim)
        mean, v_list = pca(points, dimensions)
        for i in range(0, len(v_list)):
            ax.quiver(*mean, *(mean + v_list[i]))
    except:
        mean, v_list = pca(points, dimensions)
        for i in range(0, len(v_list)):
            sum = mean + v_list[i]
            ax.arrow(*mean, *v_list[i],  head_width=3, length_includes_head=True, head_length=1)

    ax.scatter(*zip(*points))
    plt.show()
