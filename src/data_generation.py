from real_data_cleaning import clean_weather_data, is_dataframe_sorted
from sklearn.preprocessing import StandardScaler
from math import cos, sin, radians
from general_func import *
from orderings import *
import pandas as pd
import numpy as np
import logging
import random
import json
import io
from os import listdir


# Add direction as imaginary number
def add_complex_vector(sep_points, separated):
    '''Returns flattened list of points with imaginary vector to the next point'''
    new_data = []
    if separated:
        dimensions = len(sep_points[0][0])
        for line in sep_points:
            new_line = []
            for i, point in enumerate(line):
                if i != len(line) - 1:
                    new_point = [complex(point[j], line[i+1][j] - point[j]) for j in range(dimensions)]
                else:
                    new_point = [complex(point[j], 0) for j in range(dimensions)]
                new_line.append(np.array(new_point))
            new_data.append(new_line)
        new_data = flatten(new_data)

    elif not separated:
        dimensions = len(sep_points[0])
        for i, point in enumerate(sep_points):
            if i != len(sep_points) - 1:
                new_point = [complex(point[j], sep_points[i+1][j] - point[j]) for j in range(dimensions)]
            else:
                new_point = [complex(sep_points[i][j], 0) for j in range(dimensions)]
            new_data.append(new_point)
        new_data = np.array(new_data)
    return new_data


def add_id(sep_points, separated, normed= True):
    new_data = []
    def helper_add_ids(line, num_lines, num_points):
        if normed:
            range_ids = range(num_lines * num_points)
            normed_ids = [(i - np.mean(range_ids))/np.std(range_ids) for i in range_ids]
        else: 
            normed_ids = range(len(line))
        new_line = []
        for i,p in enumerate(line):
            new_p = np.append(p, normed_ids[i])
            new_line.append(new_p)
        return new_line
    
    if separated:
        for line in sep_points:
            num_lines = len(sep_points)
            num_points = len(sep_points[0])
            new_data.append(helper_add_ids(line, num_lines, num_points))
    else:
        num_points = len(sep_points)
        new_data = helper_add_ids(sep_points, 1, num_points)
    return np.array(new_data)


def get_neighbor_vector(sep_points, separated):
    new_data = []
    if not separated:
        # dimensions = len(sep_points[0])
        for i, p in enumerate(sep_points):
            if i != len(sep_points) - 1:
                new_vector = sep_points[i+1] - p
                new_data.append(new_vector)
            # else:
                # new_data.append([0 for i in range(dimensions)])
    elif separated:
        for group in sep_points:
            for i, p in enumerate(group):
                if i != len(group) - 1:
                    new_vector = group[i+1] - p
                    new_data.append(new_vector)
                # else:
                    # new_data.append([0 for i in range(dimensions)])
    return np.array(new_data)


def generate_data(bounds, dimensions, num_points, num_lines, jitter_bound):
    points = []

    for i in range(0, num_lines):
        # Generate starting points
        p_one = [random.uniform(min(bounds), max(bounds)) for i in range(0, dimensions)]
        p_two = [random.uniform(min(bounds), max(bounds)) for i in range(0, dimensions)]
        starting_points = [p_one, p_two]
        starting_points.sort()
        starting_points = [np.array(i) for i in starting_points]

        # Find point lying on a line generated by the starting points
        line_func = lambda p: starting_points[0] + p * (starting_points[1] - starting_points[0])

        rand_points = [random.random() for i in range(0, num_points)]
        rand_points.sort()
        line_points = [line_func(i) for i in rand_points]

        points.append(line_points)

    points.sort(key = lambda row: row[0][0])

    # Add jitter to points
    if jitter_bound:
        new_points = []
        for line in points:
            jitter_line = [point + [random.uniform(-abs(jitter_bound), jitter_bound) for i in range(0, dimensions)] for point in line]
            new_points.append(jitter_line)
        return new_points
    return points


def generate_higher_dimensional_spiral(n_samples, spiral_radius, spiral_height, noise_scale, n_dimensions):
    t = np.linspace(0, 2 * np.pi, n_samples)  # Parameter for the spiral
    # Initialize empty coordinate arrays for each dimension
    coordinates = []
    for _ in range(n_dimensions - 1):
        coordinates.append(np.zeros(n_samples))

    # Generate the spiral coordinates in each dimension
    for i in range(n_dimensions - 1):
        coordinates[i] = spiral_radius * np.cos((i+1) * t)
        
    # Add the height dimension
    coordinates.append(spiral_height * t)

    # Add noise to the coordinates
    for i in range(n_dimensions):
        coordinates[i] += noise_scale * np.random.randn(n_samples)

    # Stack the coordinates to form the higher-dimensional data points
    data = np.column_stack(coordinates)
    return data


def generate_spirals(num_spirals, num_points_per_spiral, spiral_radius, spiral_height, noise_factor):
    data = []
    for _ in range(num_spirals):
        spiral_data = generate_higher_dimensional_spiral(num_points_per_spiral, spiral_radius, spiral_height, noise_factor, 2)
        data.extend(spiral_data)
    return np.array(data)


def generate_swissroll(num_points, noise_factor, dim=3):
    t = np.linspace(0, 3*np.pi, num_points)
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = np.random.randn(num_points) * noise_factor
    data = np.column_stack((x, y, z))
    if dim > 3:
        extra_dim = np.random.randn(num_points, dim-3)
        data = np.column_stack((data, extra_dim))
    return data


def generate_staircase_lines(num_lines, num_points_per_line, bounds, dimensions, rotation_deg):
    
    # Initialize empty lists for x and y coordinates
    x_coords = []
    y_coords = []

    x_coords = [random.uniform(min(bounds), max(bounds)) for _ in range(num_lines)]
    x_coords.sort()
    x_coords = np.repeat(x_coords, num_points_per_line)
    
    # Generate staircase lines
    min_bound = min(bounds)
    max_bound = max(bounds) / num_lines
    for _ in range(num_lines):
        # Generate y coordinates
        y_line = [random.uniform(min_bound, max_bound) for _ in range(num_points_per_line)]
        y_line.sort()
        y_coords.extend(y_line)
        min_bound = max_bound
        max_bound += max(bounds) / num_lines
    
    # Apply rotation to the coordinates
    rotation_rad = radians(rotation_deg)
    rotated_coords = np.column_stack((x_coords, y_coords))
    rotated_coords = np.dot(rotated_coords, [[cos(rotation_rad), -sin(rotation_rad)],
                                              [sin(rotation_rad), cos(rotation_rad)]])
        
    grouped_coords = [rotated_coords[i:i + num_points_per_line] for i in range(0, num_lines * num_points_per_line, num_points_per_line)]
    
    if dimensions > 2:
        grouped_lists = []
        spare_dimensions = [[random.uniform(min_bound, max_bound) for i in range(dimensions - 2)] for _ in range(num_lines)]

        for i, grouped in enumerate(grouped_coords):
            extended = [np.hstack([p, spare_dimensions[i]]) for p in grouped]
            grouped_lists.append(extended)
        return grouped_lists
    return grouped_coords


def generate_clusters(num_clusters, num_points_per_cluster, cluster_radius, bounds, dim):
    data = []

    for i in range(num_clusters):
        center = np.random.randn(dim)
        new_point = np.random.rand(num_points_per_cluster, dim) * cluster_radius + center
        data.append(new_point)
    
    return np.array(data)


def generate_connected_zigzag(num_points_per_line, num_lines, dimensions, bounds, noise):
    points = []

    p_two = [random.uniform(min(bounds), max(bounds)) for i in range(0, dimensions)]

    for i in range(num_lines):
        # Generate random start and endpoints for each line
        p_one = p_two
        p_two = [i + random.random() for i in p_one]

        starting_points = [p_one, p_two]
        starting_points.sort()
        starting_points = [np.array(i) for i in starting_points]

        # Find point lying on a line generated by the starting points
        line_func = lambda p: starting_points[0] + p * (starting_points[1] - starting_points[0])

        rand_points = [random.random() for i in range(0, num_points_per_line)]
        rand_points.sort()
        line_points = [line_func(i) for i in rand_points]

        points.append(line_points)

    points.sort(key = lambda row: row[0][0])
    if noise:
        new_points = []
        for line in points:
            jitter_line = [point + [random.uniform(-abs(noise), noise) for i in range(0, dimensions)] for point in line]
            new_points.append(jitter_line)
        return flatten(new_points)
    
    return flatten(points)


def generate_parallel_lines(num_lines, num_points, dimensions, bounds, rotation_angle, jitter_bounds, parallel_start_end = True):
    data = []

     # Generate rotation matrix
    rotation_matrix = generate_rotation_matrix(dimensions, rotation_angle)
    
    direction = [random.random() for i in range(dimensions)]
    direction = direction / np.linalg.norm(direction)

    if parallel_start_end:
        length = random.uniform(min(bounds), max(bounds))
        start_point = np.random.rand(dimensions) * random.uniform(min(bounds), max(bounds)- length)
        starting_points = [start_point]

        for _ in range(num_lines - 1):
            dims = [random.uniform(min(bounds) + min(start_point), max(bounds) - max(start_point)) for _ in range(dimensions - 1)]
            x = [0] + dims
            new_start_point = start_point + x
            starting_points.append(new_start_point)

        for i in starting_points:
            # Generate random start and end points for each line
            end = i + direction * length

            # Rotate start and end points
            start_rotated = np.dot(rotation_matrix, i)
            end_rotated = np.dot(rotation_matrix, end)
            
            # Generate points along the line
            line_points = np.linspace(start_rotated, end_rotated, num_points)
            
            # Append the points to the data
            data.extend(line_points)
        data = [data[i: i + num_points] for i in range(0, num_points * num_lines, num_points)]
        if jitter_bounds:
            new_points = []
            for line in data:
                jitter_line = [point + [random.uniform(-abs(jitter_bounds), jitter_bounds) for i in range(0, dimensions)] for point in line]
                new_points.append(jitter_line)
            data = new_points
        return data
    
    else: 
        for _ in range(num_lines):
            # Generate random start and end points for each line
            start = np.random.rand(dimensions) * random.uniform(min(bounds), max(bounds))
            end = start + direction * random.uniform(min(bounds), (max(bounds) - max(start)))
            start_rotated = start
            end_rotated = end

            # Rotate start and end points
            start_rotated = np.dot(rotation_matrix, start)
            end_rotated = np.dot(rotation_matrix, end)
            
            # Generate points along the line
            line_points = np.linspace(start_rotated, end_rotated, num_points)
            
            # Append the points to the data
            data.extend(line_points)

        data = [data[i: i + num_points] for i in range(0, num_points * num_lines, num_points)]
        if jitter_bounds:
            new_points = []
            for line in data:
                jitter_line = [point + [random.uniform(-abs(jitter_bounds), jitter_bounds) for i in range(0, dimensions)] for point in line]
                new_points.append(jitter_line)
            data = new_points    
        return data
    

def helices(num_points, radius, height):
    radius1 = radius
    radius2 = -radius

    # Generate data points on the helices
    theta = np.linspace(0, 4 * np.pi, num_points)  # Angle parameter
    z = np.linspace(0, height, num_points)  # Height parameter

    # Generate coordinates for the first helix
    x1 = radius1 * np.cos(theta)
    y1 = radius1 * np.sin(theta)
    z1 = z

    # Generate coordinates for the second helix
    x2 = radius2 * np.cos(theta)
    y2 = radius2 * np.sin(theta)
    z2 = z

    # # Combine the coordinates of both helices
    # x = np.concatenate((x1, x2))
    # y = np.concatenate((y1, y2))
    # z = np.concatenate((z1, z2))

    # # Stack the coordinates into a single array
    # print(np.column_stack((x, y, z)))
    # return np.column_stack((x, y, z))
    helix1 = np.array([x1, y1, z1]).T
    helix2 = np.array([x2,y2,z2]).T
    
    return [helix1, helix2]
        

#TODO Fix rotation
def generate_rotation_matrix(dimensions, rotation_angle):
    # Generate an identity matrix
    rotation_matrix = np.eye(dimensions)
    
    # Compute sine and cosine of the rotation angle
    c = cos(rotation_angle)
    s = sin(rotation_angle)
    
    # Fill the rotation matrix with appropriate values
    rotation_matrix[0,0] = c
    rotation_matrix[0,1] = -s
    rotation_matrix[1,0] = s
    rotation_matrix[1,1] = c

    return rotation_matrix

def prepare_gas_data(batch_numbers):
    data_points = []
    for b in batch_numbers:
        dat_content = [i.strip().split() for i in open("./data/gas+sensor+array+drift+dataset+at+different+concentrations/batch" + str(b) + ".dat").readlines()]
        for l in dat_content:
            values = [float(pair.split(':')[1]) for pair in l[1:]]
            filtered_values = []
            
    return np.array(data_points)


def prepare_jap_air_pollution(num_points):
    logging.info("Start reading dataset")
    datasets = pd.read_csv("./data/ETL_DATA_final.csv")
    dataset = datasets[:num_points]
    for col in dataset.columns:
        dataset[col] = dataset[col].fillna(0)
    dataset = dataset.drop('TimeStamp', axis = 1)
    data = dataset.values.tolist()
    logging.info("Prepared data")
    return np.array(data)


def prepare_flight_data(num_lines, num_points_per_line):
    logging.info("Start reading dataset")
    assert num_lines <= 10, "Error: num_lines must be smaller than 10"
    data = []
    num_points = 0
    for i in range(1, num_lines + 1):
        if i < 10:
            flight_data = pd.read_csv("./data/flights/Flight00" + str(i) + ".csv")
        if i == 10:
            flight_data = pd.read_csv("./data/flights/Flight0" + str(i) + ".csv")
        assert len(flight_data) >= num_points_per_line, "Error: num_points must be smaller than " + str(len(flight_data))
        num_points += len(flight_data)
        flight_data = flight_data[: num_points_per_line]
        flight_data = flight_data.drop('time', axis = 1)
        flight_data = flight_data.values.tolist()
        data.append(np.array(flight_data))
    logging.info("Prepared data")
    return data


def prepare_russel2000_stock(num_points):
    logging.info("Start reading dataset")
    datasets = pd.read_csv("./data/russell2000data.csv")
    assert num_points <= len(datasets), "Error: Dataset only consists of 504 entries"
    dataset = datasets[:num_points]
    dataset = dataset.dropna(axis=1)
    dataset = dataset.drop('Date', axis = 1)
    print(datasets.shape)
    print(len(datasets.columns.tolist()))
    data = dataset.values.tolist()
    print(len(data[0]))
    print(len(data))
    logging.info("Prepared data")
    return np.array(data)


def prepare_cmapss_data(num_lines, num_points):
    assert num_lines <= 4, "Error: There are only 4 train datasets for CMAPSS"
    logging.info("Start reading dataset")
    files = [f for f in listdir("data/CMAPSS/")][:num_lines]
    data = []
    for f in files:
        content = []
        with io.open("data/CMAPSS/" + str(f)) as file:
            for i, line in enumerate(file):
                if i < num_points:
                    points = [float(p) for p in line.split()]
                    content.append(points[2:])
        data.append(content)
    logging.info("Prepared data")
    return np.array(data)


def generate_correct_ordered_data(order, num_points_per_line, num_lines, start_dim, jitter_bound, bounds, parallel_start_end = False, standardise = False):
    separated = False
    if order == "weather":
        weather_data = pd.read_csv("./data/weather_data.csv")
        weather_data["utc_timestamp"] = pd.to_datetime(weather_data["utc_timestamp"])
        if not is_dataframe_sorted(weather_data, "utc_timestamp"):
            weather_data = weather_data.sort_values(by="utc_timestamp")
        weather_data = weather_data.drop("utc_timestamp", axis = 1)
        weather_data.dropna()
        sep_points = clean_weather_data(weather_data)
        num_lines = 1
        num_points_per_line = len(sep_points)
        start_dim = len(sep_points[0])
    elif order == "gas":
        sep_points = prepare_gas_data(list(range(1, 10)))
        num_lines = len(sep_points)
        num_points_per_line = len(sep_points[0])
        start_dim = len(sep_points[0][0])
    elif order == "air_pollution":
        sep_points = prepare_jap_air_pollution(num_points_per_line)
        num_lines = 1
        start_dim = len(sep_points[0])
    elif order == "russell2000_stock":
        sep_points = prepare_russel2000_stock(num_points_per_line)
        num_lines = 1
        start_dim = len(sep_points[0])
    elif order == "flights":
        separated = True
        sep_points = prepare_flight_data(num_lines, num_points_per_line)
        start_dim = len(sep_points[0][0])
    elif order == "CMAPSS":
        separated = True
        sep_points = prepare_cmapss_data(num_lines, num_points_per_line)
        start_dim = sep_points[0][0]
    elif order == "spiral":
        spiral_height = 10
        sep_points = generate_higher_dimensional_spiral(num_points_per_line, 100, spiral_height, jitter_bound, start_dim)
    elif order == "swiss_roll":
        sep_points = generate_swissroll(num_points_per_line, jitter_bound, start_dim)
    elif order == "staircase":
        separated = True
        sep_points = generate_staircase_lines(num_lines, num_points_per_line, bounds, start_dim, rotation_deg = 90)
    elif order == "connected_staircase":
        sep_points = generate_parallel_lines(num_lines, num_points_per_line, start_dim, bounds, 0, jitter_bound, parallel_start_end=parallel_start_end)
        sep_points = flatten(sep_points)
    elif order == "parallel_lines":
        sep_points = generate_parallel_lines(num_lines, num_points_per_line, start_dim, bounds, 0, jitter_bound, parallel_start_end=parallel_start_end)
        separated = True
    elif order == "connected_parallel_lines":
        sep_points = generate_parallel_lines(num_lines, num_points_per_line, start_dim, bounds, 0, jitter_bound, parallel_start_end=parallel_start_end)
        sep_points = flatten(sep_points)
    elif order == "helices":
        sep_points = helices(num_points_per_line, 1, spiral_height)
    elif order == "connected_zigzag":
        sep_points = generate_connected_zigzag(num_points_per_line, num_lines, start_dim, bounds, jitter_bound)
    elif order == "clusters":
        sep_points = generate_clusters(num_lines, num_points_per_line, 1, bounds, start_dim)
        separated = True
    else:
        sep_points = generate_data(bounds, start_dim, num_points_per_line, num_lines, jitter_bound)
        sep_points = order_nearest_start_points(sep_points)
        
        if order == "zigzag":
            sep_points = zigzag_order(sep_points, num_points_per_line)
            separated = False
        elif order == "one_line":
            sep_points = one_line(sep_points)
            separated = False
        elif order == "random":
            sep_points = random_ordering(flatten(sep_points))
            separated = False
        elif order == "sep_lines":
            separated = True
        elif order == "spatial":
            sep_points = sort_nearest_neighbour(flatten(sep_points))
        
    if standardise:
        object = StandardScaler()
        if len(sep_points) == num_lines * num_points_per_line or len(sep_points) == num_points_per_line:
            flat_points = sep_points
        else:
            flat_points = flatten(sep_points)
            sep_points = object.fit_transform(flat_points)
            sep_points = list([list(sep_points[i:i + num_points_per_line]) for i in range(0, len(sep_points), num_points_per_line)])
            
    return sep_points, separated

def get_picked_options():
    with open('src/config.json') as json_file:
        config = json.load(json_file)
    
    num_points_per_line = config["num_points_per_line"]
    num_lines = config["num_lines"]
    starting_dimension = config["starting_dimension"]
    target_dimension = config["target_dimension"]
    bounds = config["bounds"]
    seed = config["seed"]
    jitter = config["jitter_bound"]
    parallel_start_end = config["parallel_start_end"]
    sep_measure = config["sep_measure"]
        
    sep_points, separated = generate_correct_ordered_data(config["order"], num_points_per_line, num_lines, starting_dimension, config["jitter_bound"], config["bounds"], config["parallel_start_end"])
    
    if config["order"] in ["air_pollution", "russell2000_stock"]:
        num_points_per_line = len(sep_points)
        num_lines = 1
        starting_dimension = len(sep_points[0])
        bounds = [min(flatten(sep_points)), max(flatten(sep_points))]
        seed = 0
        jitter = 0
        parallel_start_end = False
        sep_measure = False
    elif config["order"] in ["flights", "CMAPSS"]:
        starting_dimension = len(sep_points[0][0])
        bounds = [min(flatten(flatten(sep_points))), max(flatten(flatten(sep_points)))]
        seed = 0
        jitter = 0
        parallel_start_end = False
        sep_measure = True
    
    picked_options = { # Do not change these!
        "seed" : seed,
        "sep_points" : sep_points,
        "order" : config["order"],
        "separated": separated,
        "starting_dimension" : starting_dimension,
        "target_dimension" : target_dimension,
        "pca_choice" : config["pca_choice"],
        "scaling_choice": config["scaling_choice"],
        "when_shuffle" : config["when_shuffle"],
        "rec_error_real" : config["reconstruction_error_real"],
        "num_lines": num_lines,
        "num_points_per_line": num_points_per_line,
        "bounds": bounds,
        "standardised": config["standardise"],
        "compare_k_neighbors" : config["k"],
        "sep_measure": sep_measure,
        "dist_measure": config["dist_measure"],
        "jitter_bound" : jitter,
        "parallel_start_end" : parallel_start_end,
        "outliers": [[random.uniform(min(config["bounds"]), max(config["bounds"])) for _ in range(config["starting_dimension"])] for _ in range(config["num_outliers"])]
    }
    return picked_options

get_picked_options()