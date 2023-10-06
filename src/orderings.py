import general_func
import numpy as np
import random
import math
from sklearn.neighbors import NearestNeighbors

def order_nearest_start_points(sep_points):
    try:
        borders = [[[i, line[0]], [i, line[-1]]] for i, line in enumerate(sep_points)]
        border_points = [item for i in borders for item in i]
        point1 = []
        min_diff = math.inf
        for i in range(len(border_points)):
            j = i + 1
            while j < len(border_points):
                diff = math.dist(border_points[i][1], border_points[j][1])
                if diff < min_diff and border_points[i][0] != border_points[j][0]:
                    point1 = border_points[i]
                    min_diff = diff
                j += 1
        new_order = []
        for line in sep_points:
            border1 = line[0]
            border2 = line[-1]
            diff1 = math.dist(border1, point1[1])
            diff2 = math.dist(border2, point1[1])
            if diff1 < diff2:
                new_order.append(line)
            else:
                new_order.append(line[::-1])
        return new_order
    except:
        return sep_points


# Takes everytime the first point
def zigzag_order(points, num_points):
    zigzag = []
    for i in range(0, num_points):
        ith_points = [line[i] for line in points]
        zigzag.append(ith_points)
    # return flat points
    return general_func.flatten(zigzag)


# Strict zigzag order
def line_zigzag_order(sep_points, num_points):
    sep_points = order_nearest_start_points(sep_points)

    flat_points_w_order = [(i, item) for (i, line) in enumerate(sep_points) for item in line]
    # TODO sorting wrong
    next_point = flat_points_w_order.pop(0)
    new_list = [next_point[1]]
    while len(flat_points_w_order) > 0:
        # next_point = next(filter(lambda k: k[0] != next_point[0], flat_points_w_order), None)
        min_diff = math.inf
        for i, point in enumerate(flat_points_w_order):
            diff = math.dist(next_point[1], point[1])
            if diff < min_diff and next_point[0] != point[0]:
                min_diff = diff
                next_point = flat_points_w_order.pop(i)
                new_list.append(next_point[1])
            elif len(set([i[0] for i in flat_points_w_order])) == 1:
                new_list = new_list + [i[1] for i in flat_points_w_order]
                flat_points_w_order = []
    return np.array(new_list)

   
# Connect all points with one line
def one_line(sep_points):
    sep_points = order_nearest_start_points(sep_points)
    one_line_list = []
    for i, point_line in enumerate(sep_points):
        if i % 2 == 1:
            point_line.reverse()
        one_line_list.append(np.array(point_line))
    return general_func.flatten(one_line_list)


def sort_nearest_neighbour(data):
    # Create a NearestNeighbors instance with k = 2
    k = 2
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(data)
    
    # Get the indices of the nearest neighbors for each data point
    _, indices = nn.kneighbors(data)
    
    # Initialize an array to store the ordered data
    ordered_data = np.empty_like(data)
    
    # Reorder the data based on the nearest neighbors
    for i, neighbor_indices in enumerate(indices):
        ordered_data[i] = data[neighbor_indices[0]]
    return ordered_data


# Reverse given ordering to test pca
def reverse_ordering(flat_points):
    ''' Get flat points and reverse order'''
    return(np.flip(flat_points, 0))


def random_ordering(flatpoints):
    num_points = len(flatpoints)
    dimension = len(flatpoints[0])
    random_data = np.random.uniform(0, 100, size=(num_points, dimension))
    return random_data

def shuffle_under_seed(ls, seed):
  # Shuffle the list ls using the seed `seed`
  random.seed(seed)
  random.shuffle(ls)
  return ls

def unshuffle_list(shuffled_ls, seed):
  n = len(shuffled_ls)
  # Perm is [1, 2, ..., n]
  perm = [i for i in range(1, n + 1)]
  # Apply sigma to perm
  shuffled_perm = shuffle_under_seed(perm, seed)
  # Zip and unshuffle
  zipped_ls = list(zip(shuffled_ls, shuffled_perm))
  zipped_ls.sort(key=lambda x: x[1])
  return [a for (a, b) in zipped_ls]

def insert_point(points, point, after_point = None, before_point = None):
    '''Assumption: All points are unique. \\
        Add None for after_point if new point is to be inserted at start \\'''
    #find out if points are all connected or not
    seperated = False
    if type(points[0]) == list:
        seperated == True

    index_after_p = None
    index_before_p = None

    if seperated:
        for line in points:
            for i,p in enumerate(line):
                if type(after_point) != type(None):
                    if [x.real for x in p] == [x.real for x in after_point]:
                        index_after_p = i
                if type(before_point) != type(None):
                    if [x.real for x in p] == [x.real for x in before_point]:
                        index_before_p = i
            if index_after_p and index_before_p:
                assert(index_after_p + 1 == index_before_p)
                real_after_point = [x.real for x in after_point]
                real_before_point = [x.real for x in before_point]
                new_complex_after = [complex(real_after_point[i], point - real_after_point[i]) for i in range(len(after_point))]
                complex_point = [complex(point[i], real_before_point[i] - point [i]) for i in range(len(after_point))]
                line[index_after_p] = new_complex_after
                np.insert(line, index_after_p + 1, np.array(complex_point), axis = 0)
            elif index_after_p:
                real_after_point = [x.real for x in after_point]
                new_complex_after = [complex(real_after_point[i], point - real_after_point[i]) for i in range(len(after_point))]
                next_point = line[index_after_p + 1]
                real_next_point = [x.real for x in next_point]
                complex_point = [complex(point[i], real_next_point[i] - point [i]) for i in range(len(after_point))]
                line[index_after_p] = new_complex_after
                np.insert(line, index_after_p + 1, np.array(complex_point), axis = 0)
            elif index_before_p:
                if index_before_p == 0:
                    real_before_point = [x.real for x in before_point]
                    complex_point = [complex(point[i], real_next_point[i] - point [i]) for i in range(len(before_point))]
                    np.insert(line, index_before_p, np.array(complex_point), axis = 0)
                else:
                    point_before = line[index_before_p - 1]
                    real_point_before = [x.real for x in point_before]
                    new_complex_point_before = [complex(real_point_before[i], point[i] - real_point_before[i]) for i in range(len(point_before))]
                    real_before_point = [x.real for x in before_point]
                    complex_point = [complex(point[i], real_before_point[i] - point[i]) for i in range(len(point))]
                    line[index_before_p - 1] = new_complex_point_before
                    np.insert(line, index_before_p, np.array(complex_point), axis = 0)
    else:
        for i,p in enumerate(points):
                if type(after_point) != type(None):
                    if [x.real for x in p] == [x.real for x in after_point]:
                        index_after_p = i
                if type(before_point) != type(None):
                    if [x.real for x in p] == [x.real for x in before_point]:
                        index_before_p = i
        if index_after_p and index_before_p:
            assert(index_after_p + 1 == index_before_p)
            real_after_point = [x.real for x in after_point]
            real_before_point = [x.real for x in before_point]
            new_complex_after = [complex(real_after_point[i], point[i] - real_after_point[i]) for i in range(len(after_point))]
            complex_point = [complex(point[i], real_before_point[i] - point[i]) for i in range(len(after_point))]
            points[index_after_p] = new_complex_after
            np.insert(points, index_after_p + 1, np.array(complex_point), axis = 0)
        elif index_after_p:
            real_after_point = [x.real for x in after_point]
            new_complex_after = [complex(real_after_point[i], point[i] - real_after_point[i]) for i in range(len(after_point))]
            next_point = points[index_after_p + 1]
            real_next_point = [x.real for x in next_point]
            complex_point = [complex(point[i], real_next_point[i] - point[i]) for i in range(len(after_point))]
            points[index_after_p] = 5
            np.insert(points, index_after_p + 1, np.array(complex_point), axis = 0)
        elif index_before_p:
            if index_before_p == 0:
                real_before_point = [x.real for x in before_point]
                complex_point = [complex(point[i], real_next_point[i] - point[i]) for i in range(len(before_point))]
                np.insert(points, index_before_p, np.array(complex_point), axis = 0)
            else:
                point_before = points[index_before_p - 1]
                real_point_before = [x.real for x in point_before]
                new_complex_point_before = [complex(real_point_before[i], point[i] - real_point_before[i]) for i in range(len(point_before))]
                real_before_point = [x.real for x in before_point]
                complex_point = [complex(point[i], real_before_point[i] - point[i]) for i in range(len(point))]
                points[index_before_p - 1] = new_complex_point_before
                np.insert(points, index_before_p, np.array(complex_point), axis = 0)

